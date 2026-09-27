"""Bounded owned-process supervisor and single frozen ABBA context pilot."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def start_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def stop_owned(process, ticks):
    """Signal only the dedicated session/group created for this exact Popen."""
    cleanup = {"pid": process.pid, "signals": []}
    current = start_ticks(process.pid)
    if current is not None and current != ticks:
        cleanup["refused"] = "PID identifier reused"
        return cleanup
    for sig, grace in [(signal.SIGTERM, 1.0), (signal.SIGKILL, 0.5)]:
        try:
            os.killpg(process.pid, sig)
            cleanup["signals"].append(sig.name)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
        # The leader may have exited while its child processes still hold the
        # same dedicated group; kill that group after the short grace below.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            time.sleep(0.1)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=0.5)
    cleanup["exit_code"] = process.returncode
    return cleanup


def run_owned_trial(
    out, server_cmd, client_cmd, env, cwd, deadline, *, startup_limit=180, poll=0.2
):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trial-status.json").open("x") as f:
        f.write("{}\n")
    processes = []
    result = {
        "status": "starting",
        "started_at": datetime.now().astimezone().isoformat(),
        "pids": {},
        "cleanup": [],
    }
    started = time.monotonic()
    try:
        with contextlib.ExitStack() as stack:

            def spawn(name, command):
                log = stack.enter_context((out / (name + ".log")).open("x"))
                p = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                ticks = start_ticks(p.pid)
                assert ticks is not None and os.getpgid(p.pid) == p.pid
                processes.append((p, ticks))
                result["pids"][name] = p.pid
                return p

            server = spawn("server", server_cmd)
            startup_deadline = min(deadline, time.monotonic() + startup_limit)
            while not (out / "server-ready.json").exists():
                if server.poll() is not None:
                    raise RuntimeError(
                        f"server exited before ready: {server.returncode}"
                    )
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError("server startup deadline")
                time.sleep(poll)
            ready = json.loads((out / "server-ready.json").read_text())
            assert ready["model_initialized"]
            if time.monotonic() >= deadline:
                raise TimeoutError("trial deadline before client")
            result["server_ready_seconds"] = time.monotonic() - started
            client = spawn("client", client_cmd)
            result["status"] = "running"
            write_json(out / "trial-status.json", result)
            while client.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"server exited during client: {server.returncode}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("client/global work deadline")
                time.sleep(poll)
            result["client_exit_code"] = client.returncode
            if client.returncode != 0:
                raise RuntimeError(f"client exited: {client.returncode}")
            result["status"] = "completed"
    except BaseException as error:
        result.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
    finally:
        for p, ticks in reversed(processes):
            result["cleanup"].append(stop_owned(p, ticks))
        result["elapsed_seconds"] = time.monotonic() - started
        result["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(out / "trial-status.json", result)
    return result


def preflight(out, gpu, deadline):
    samples = []
    for i in range(3):
        if time.monotonic() + 1 >= deadline:
            raise TimeoutError("preflight deadline")
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        memory, util, ecc = map(int, raw.split(","))
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        samples.append(
            {
                "memory_mib": memory,
                "utilization": util,
                "ecc": ecc,
                "compute_process_present": gpu in apps,
            }
        )
        write_json(out / "preflight.json", samples)
        if memory >= 100 or util != 0 or ecc != 0 or gpu in apps:
            raise RuntimeError("GPU is not idle; no process is stopped")
        if i < 2:
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 18848))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--study", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    study = a.study.resolve()
    protocol = json.loads((study / "run-plan.json").read_text())
    assert protocol["wall_limit_seconds"] == 600 and [
        (t["id"], t["mode"]) for t in protocol["trials"]
    ] == [
        ("A0", "resident_idle"),
        ("B0", "shadow_online"),
        ("B1", "shadow_online"),
        ("A1", "resident_idle"),
    ]
    manifest = json.loads((study / "source-manifest.json").read_text())
    for relative, digest in manifest["runtime_files"].items():
        if file_hash(root / relative) != digest:
            raise RuntimeError("source changed: " + relative)
    for relative, digest in manifest["study_files"].items():
        if file_hash(study / relative) != digest:
            raise RuntimeError("study input changed: " + relative)
    lock = (root / "gpu-cm001-2.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    deadline = started + 600
    launch = {
        "started_at": datetime.now().astimezone().isoformat(),
        "wall_limit_seconds": 600,
        "work_budget_seconds": 595,
        "gpu": protocol["gpu"],
        "trials": [],
    }
    with (study / "launch.json").open("x") as f:
        json.dump(launch, f, indent=2)

    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=protocol["gpu"],
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        HF_HOME=str(root / "cache/huggingface"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TORCHINDUCTOR_CACHE_DIR=str(root / "cache/torchinductor"),
        TORCH_EXTENSIONS_DIR=str(root / "cache/torch_extensions"),
        ROBOTWIN_PATH=str(root / "benchmarks/RoboTwin"),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(
            root
            / "benchmark-env/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
        ),
        PYTHONPATH=str(root / "benchmarks/RoboTwin")
        + ":"
        + str(root / "OpenWAM/benchmarks/robotwin"),
        MPLCONFIGDIR=str(root / "cache/matplotlib"),
        ROBUST_CONDITION="clean",
        NORM_SIGMA="0",
        NORM_ROLE="context_fixed_command_diagnostic",
        NORM_ARM="identity",
        NORM_SCENES=str(study / "scenes.json"),
        ROBOTWIN_TEST_NUM="1",
        CONTEXT_STUDY=str(study),
    )
    code = 1
    try:
        for position, spec in enumerate(protocol["trials"]):
            out = study / spec["id"]
            out.mkdir()
            preflight(out, protocol["gpu"], deadline - 5)
            env.update(
                CONTEXT_MODE=spec["mode"],
                ROBUST_RUN_DIR=str(out),
                ROBOTWIN_RUNTIME_ROOT=str(out / "runtime"),
            )
            server = [
                str(root / "policy-env/bin/python"),
                "-u",
                str(root / "scripts/serve_context.py"),
                "--out",
                str(out),
                "--port",
                "18848",
            ]
            client = [
                str(root / "benchmark-env/bin/python"),
                "-u",
                str(root / "scripts/context_client.py"),
                "--config",
                str(root / "OpenWAM/benchmarks/robotwin/policy_config.yml"),
                "--overrides",
                "--task_name",
                "pick_dual_bottles",
                "--task_config",
                "demo_clean",
                "--ckpt_setting",
                "context-" + spec["id"],
                "--seed",
                "3",
                "--policy_name",
                "openwam2robotwin_interface",
                "--host",
                "127.0.0.1",
                "--port",
                "18848",
            ]
            trial = run_owned_trial(out, server, client, env, root, deadline - 5)
            launch["trials"].append({"id": spec["id"], "mode": spec["mode"], **trial})
            write_json(study / "launch.json", launch)
            print("CONTEXT_TRIAL", spec["id"], trial["status"], flush=True)
            if trial["status"] != "completed":
                raise RuntimeError("trial failed; remaining trials not substituted")
            if position < 3:
                if time.monotonic() + 2 >= deadline - 5:
                    raise TimeoutError("insufficient budget for next trial")
                time.sleep(2)
        code = 0
    except BaseException as error:
        launch.update(error_type=type(error).__name__, error=str(error))
    finally:
        launch.update(
            exit_code=code,
            elapsed_seconds=time.monotonic() - started,
            finished_at=datetime.now().astimezone().isoformat(),
        )
        write_json(study / "launch.json", launch)
        (study / "exit-code.txt").write_text(str(code) + "\n")
        lock.close()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
