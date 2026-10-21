"""Prepared GPU probe; refuses all unregistered invocations. Not yet dispatched."""
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
from adapter import guard, verify_device

R=Path("/home/zifanz4/openwam-runtime")
ROOT=R/"ex8-a_officialrepro_1021"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run(registration, output):
    p=json.loads(registration.read_text())
    assert p.get("kind")=="FROZEN_NATIVE_RENDER_PROBE" and p.get("registered_for_dispatch") is True
    assert 0<p["max_seconds"]<=180
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
    own_group=Path("/proc/self/cgroup").read_text()
    def observe():
        while not done.is_set():
            try:
                rows=inventory.probe();contexts=[]
                for gpu in rows["gpus"]:
                    for process in gpu["processes"]:
                        try:owned=Path(f"/proc/{process['pid']}/cgroup").read_text()==own_group
                        except FileNotFoundError:continue
                        if owned:contexts.append(process)
                samples.append({"time":rows["time"],"contexts":contexts})
                if any(c["uuid"]!=assigned for c in contexts):raise RuntimeError("Peer GPU context observed")
                for path in p["pause_paths"]:assert not Path(path).exists(),path
            except BaseException as exc:errors.append(repr(exc));done.set();return
            done.wait(.1)
    def timeout(*_):raise TimeoutError("Bounded renderer probe expired")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p["max_seconds"]-15)
    monitor=threading.Thread(target=observe,daemon=True);monitor.start()
    result={"status":"running","scored_episodes":0,"full_paper_reproduced":False}
    try:
        wrapper=load("frozen_native_wrapper",ROOT/"official/benchmarks/robotwin/eval_policy_wrapper.py")
        # Match official CUDA-before-SAPIEN order. No task or policy inference.
        wrapper._prewarm_cuda_for_curobo()
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        import sapien
        anchor=sapien.render.RenderSystem(device="cuda:0")
        result["explicit_device"]=verify_device(anchor.device,expected)
        engine=sapien.Engine();engine.set_renderer(sapien.SapienRenderer())
        scene=engine.create_scene()
        result["default_scene_device"]=verify_device(scene.render_system.device,expected)
        for _ in range(15):
            if done.wait(1):break
        assert not errors and len(samples)>=3
        assert any(c["uuid"]==assigned for s in samples for c in s["contexts"])
        result["status"]="renderer_probe_passed"
    except BaseException as exc:
        result.update(status="failed",error=repr(exc));raise
    finally:
        done.set();monitor.join(timeout=10);signal.alarm(0)
        result.update(samples=samples,monitor_errors=errors,occupied_seconds=time.monotonic()-start)
        assert not Path("/home/zifanz4/.local/state/openwam-selfcheck/paused.json").exists()
        output.write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--registration",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    a=parser.parse_args();run(a.registration,a.output)
