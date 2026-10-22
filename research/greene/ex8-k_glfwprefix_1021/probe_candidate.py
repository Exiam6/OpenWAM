"""Frozen renderer-only probe; no native task or score."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from adapter import guard, verify_device, pci
from prewarm import checked_prewarm

R=Path("/home/zifanz4/openwam-runtime")
ROOT=R/"ex8-a_officialrepro_1021"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run(registration, output):
    p=json.loads(registration.read_text())
    assert p.get("kind")=="FROZEN_NATIVE_GLFW_PREFIX" and p.get("registered_for_dispatch") is True
    assert p["max_seconds"]==100 and p["probe_alarm_seconds"]==75
    assert not output.exists()
    now=dt.datetime.now(dt.timezone.utc)
    assert now<dt.datetime.fromisoformat(p["deadline"])
    assert now<dt.datetime.fromisoformat(os.environ["OPENWAM_ALLOCATION_END"])-dt.timedelta(seconds=p["max_seconds"])
    assert str(Path(__file__).resolve()) in p["input_sha256"]
    for name,digest in p["input_sha256"].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    for path in p["pause_paths"]:assert not Path(path).exists(),path
    assigned=guard(os.environ,["--config","renderer-only-probe.yml"])
    bus=subprocess.run(["nvidia-smi","--query-gpu=uuid,pci.bus_id","--format=csv,noheader"],check=True,capture_output=True,text=True,timeout=8)
    expected=dict(line.strip().split(", ") for line in bus.stdout.splitlines())[assigned]
    inventory=load("frozen_probe",R/"nightly-gpu/code-v3/probe.py")
    samples=[];errors=[];done=threading.Event();start=time.monotonic()
    phase="before_prewarm";phases=[]
    own_group=Path("/proc/self/cgroup").read_text()
    assert "openwam-nightly-job-" in own_group, "Dispatcher systemd unit required"
    def observe():
        while not done.is_set():
            try:
                sample_phase=phase
                rows=inventory.probe();contexts=[]
                for gpu in rows["gpus"]:
                    for process in gpu["processes"]:
                        try:owned=Path(f"/proc/{process['pid']}/cgroup").read_text()==own_group
                        except FileNotFoundError:continue
                        if owned:contexts.append(process)
                samples.append({"time":rows["time"],"contexts":contexts,"phase":sample_phase})
                if any(c["uuid"]!=assigned for c in contexts):raise RuntimeError("Peer GPU context observed")
                for path in p["pause_paths"]:assert not Path(path).exists(),path
            except BaseException as exc:
                errors.append(repr(exc));done.set()
                os.kill(os.getpid(),signal.SIGUSR1)
                return
            done.wait(.1)
    def peer_stop(*_):raise RuntimeError("Context monitor requested immediate own-process stop")
    signal.signal(signal.SIGUSR1,peer_stop)
    def timeout(*_):raise TimeoutError("Bounded renderer probe expired")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p["probe_alarm_seconds"])
    monitor=threading.Thread(target=observe,daemon=True);monitor.start()
    result={"status":"running","scored_episodes":0,"full_paper_reproduced":False}
    def checkpoint(name):
        nonlocal phase
        phase=name
        phases.append(dict(phase=name,elapsed_seconds=time.monotonic()-start))
        deadline=time.monotonic()+15
        while sum(s["phase"]==name for s in samples)<3:
            if done.wait(.1):raise RuntimeError("Context monitor stopped at "+name)
            if time.monotonic()>deadline:raise TimeoutError("Missing complete phase samples at "+name)
        assert not errors,errors
    try:
        checkpoint("before_prewarm")
        phase="native_prewarm"
        wrapper=load("frozen_native_wrapper",ROOT/"official/benchmarks/robotwin/eval_policy_wrapper.py")
        # Match official CUDA-before-SAPIEN order. No task or policy inference.
        result["prewarm_stdout"]=checked_prewarm(wrapper._prewarm_cuda_for_curobo)
        checkpoint("after_prewarm")
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        warm_tensor=torch.zeros(1,device="cuda")
        phase="sapien_import"
        import sapien
        checkpoint("after_sapien_import")
        phase="explicit_Device"
        device=sapien.Device("cuda:0")
        result["discovered_device"]=verify_device(device,expected)
        checkpoint("after_explicit_Device")
        phase="helper_load"
        import ctypes
        lib=ctypes.CDLL(str(registration.parent/"code/libglfw_prefix.so"))
        lib.glfw_prefix_error.restype=ctypes.c_char_p
        lib.glfw_prefix_init.restype=ctypes.c_int
        lib.glfw_prefix_supported.restype=ctypes.c_int
        checkpoint("after_helper_load")
        phase="native_glfwInit"
        result["glfw_init"]=lib.glfw_prefix_init()
        assert result["glfw_init"]==1,lib.glfw_prefix_error()
        checkpoint("after_native_glfwInit")
        phase="native_glfwVulkanSupported"
        result["glfw_vulkan_supported"]=lib.glfw_prefix_supported()
        assert result["glfw_vulkan_supported"]==1,lib.glfw_prefix_error()
        checkpoint("after_native_glfwVulkanSupported")
        result["explicit_device"]=verify_device(device,expected)
        for _ in range(3):
            if done.wait(1):break
        assert not errors and len(samples)>=3
        assert any(c["uuid"]==assigned for s in samples for c in s["contexts"])
        result["status"]="glfw_prefix_passed"
    except BaseException as exc:
        result.update(status="failed",error=repr(exc));raise
    finally:
        done.set();monitor.join(timeout=10);signal.alarm(0)
        result.update(samples=samples,phases=phases,failed_or_final_phase=phase,monitor_errors=errors,occupied_seconds=time.monotonic()-start)
        assert not Path("/home/zifanz4/.local/state/openwam-selfcheck/paused.json").exists()
        if errors:result.update(status="failed",error="Context monitor failed")
        output.write_text(json.dumps(result,indent=2)+"\n")
        if errors:raise RuntimeError("Context monitor failed")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--registration",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    a=parser.parse_args();run(a.registration,a.output)
