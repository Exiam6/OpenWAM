"""One new-task transfer pilot, with a fixed cohort and retained failures."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

ROOT = Path("/home/zifanz4/openwam-runtime/ex6-a_tasktransfer_1020")
RUNTIME = ROOT.parent
SOURCE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def admission(task):
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert task in protocol["tasks"]
    assert not (ROOT / "tasks" / task).exists(), "An attempt already exists; no retries"
    assert not (ROOT / "paused.json").exists()
    assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(protocol["deadline"])
    assert shutil.disk_usage(ROOT).free > 100 * (1 << 30)
    for path, digest in protocol["code_sha256"].items():
        assert sha(path) == digest, path
    for arm in protocol["arms"]:
        assert Path(arm["checkpoint"]).is_file()
    return protocol


def stop(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def main(task):
    p = admission(task)
    out = ROOT / "tasks" / task
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {"status": "preparing", "task": task, "started": dt.datetime.now(dt.timezone.utc).isoformat(),
             "training_seed": 42, "interpretation": p["interpretation"], "groups": []}
    server = child = None
    end = min(dt.datetime.fromisoformat(os.environ["OPENWAM_ALLOCATION_END"]),
              dt.datetime.fromisoformat(p["deadline"]))
    def save():
        write(out / "result.json", dict(state, occupied_seconds=time.monotonic()-started))
    def guard():
        assert time.monotonic()-started < p["job_max_seconds"]-180
        assert dt.datetime.now(dt.timezone.utc) < end-dt.timedelta(seconds=120)
        assert not (ROOT / "paused.json").exists()
        assert shutil.disk_usage(ROOT).free > 60 * (1 << 30)
        if server is not None:
            assert server.poll() is None, "Policy server exited"
    def terminated(*_):
        raise RuntimeError("Task terminated; partial attempt preserved")
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    save()
    try:
        code = out / "code"
        shutil.copytree(ROOT / "template", code)
        (code / "scenes").mkdir()
        gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        pci = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            text=True, timeout=15).strip()
        env = json.loads((ROOT / "base-environment.json").read_text())
        env.update(CUDA_VISIBLE_DEVICES=gpu, EXPECTED_RENDER_PCI=pci,
                   ROBOT_CONFIG_ROOT=str(ROOT / "robot-configs"), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=p["engine"],
                   PATH=f"{RUNTIME}/policy-env/bin:{RUNTIME}/cuda-toolkit/bin:/usr/bin:/bin")
        simenv = dict(env, ROBOTWIN_RUNTIME_ROOT=str(out / "simulator-runtime"),
                      PATH=f"{RUNTIME}/benchmark-env/bin:{RUNTIME}/cuda-toolkit/bin:/usr/bin:/bin")
        simenv.pop("PYTHONPATH", None)
        write(out / "environment.json", env)
        write(out / "code-freeze.json", {str(x): sha(x) for x in code.iterdir() if x.is_file()})
        def execute(argv, log, cap):
            nonlocal child
            guard()
            with log.open("x") as handle:
                child = subprocess.Popen(argv, env=simenv, cwd=p["engine"], stdin=subprocess.DEVNULL,
                                         stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            tick = time.monotonic()
            try:
                while child.poll() is None:
                    guard()
                    assert time.monotonic()-tick < cap, f"Child exceeded {cap} seconds"
                    time.sleep(2)
                assert child.returncode == 0, f"Child failed; inspect {log}"
            finally:
                stop(child)
                child = None
        state["status"] = "expert_cohort"
        save()
        execute([str(RUNTIME / "benchmark-env/bin/python"), "-u", str(code / "collect_pilot.py"),
                 "--task", task, "--code", str(code)], out / "collection.log", 2800)
        manifest = code / "scenes" / task / "manifest.json"
        scenes = [r["seed"] for r in json.loads(manifest.read_text())["accepted"]]
        state["cohort_sha256"] = sha(manifest)
        state["scenes"] = scenes
        # Hash all inputs before loading; checkpoint identity is part of the frozen protocol.
        for arm in p["arms"]:
            guard()
            assert sha(arm["checkpoint"]) == arm["sha256"], arm["name"]
        for arm in p["arms"]:
            route = arm["name"]
            dest = out / route
            dest.mkdir()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            state.update(status="loading_policy", active_arm=route)
            save()
            with (dest / "server.log").open("x") as log:
                server = subprocess.Popen([str(RUNTIME / "policy-env/bin/python"), "-u",
                    str(code / "serve_policy.py"), "--checkpoint", arm["checkpoint"],
                    "--port", str(port), "--log", str(dest / "episode-rng.jsonl")],
                    env=env, cwd=p["engine"], stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            tick = time.monotonic()
            while "ENDTOEND_SERVER_READY" not in (dest / "server.log").read_text(errors="replace"):
                guard()
                assert time.monotonic()-tick < 900, "Policy loading timeout"
                time.sleep(2)
            # Use known training scenes for transport/reset smoke, never scored new scenes.
            smoke = dest / "smoke"
            smoke.mkdir()
            execute([str(RUNTIME / "benchmark-env/bin/python"), "-u", str(code / "simulation_smoke.py"),
                     "--port", str(port), "--output", str(smoke)], dest / "smoke.log", 600)
            assert json.loads((smoke / "result.json").read_text())["passed"]
            hashes = {str(x): sha(x) for x in code.iterdir()
                      if x.is_file() and not x.name.startswith("simulation-preflight-")}
            write(code / f"simulation-preflight-{route}-42.json",
                  {"passed": True, "code_sha256": hashes, "smoke": str(smoke / "result.json")})
            state["status"] = "closed_loop"
            save()
            episodes = dest / "episodes"
            episodes.mkdir()
            execute([str(RUNTIME / "benchmark-env/bin/python"), "-u", str(code / "evaluate_policy.py"),
                     "--task", task, "--route", route, "--seed", "42", "--condition", "clean",
                     "--scene-seeds", json.dumps(scenes), "--port", str(port), "--output", str(episodes)],
                    dest / "evaluation.log", 3150)
            summary = json.loads((episodes / "summary.json").read_text())
            state["groups"].append({"arm": route, "episodes": summary["episodes"],
                                    "successes": summary["successes"],
                                    "technical_failures": summary["technical_failures"],
                                    "summary": str(episodes / "summary.json")})
            save()
            assert summary["complete"] and summary["technical_failures"] == 0
            stop(server)
            server = None
        # Ensure the same initial state, prompt and action RNG across every pair.
        for scene in scenes:
            rows = [json.loads((out / a["name"] / "episodes" / f"seed-{scene}" / "result.json").read_text())
                    for a in p["arms"]]
            for key in ["initial_state", "evaluation_instruction", "action_seed", "step_limit"]:
                assert len({json.dumps(row[key], sort_keys=True) for row in rows}) == 1, key
        state.update(status="complete", complete=True, paired_initial_conditions_verified=True)
    except BaseException as exc:
        state.update(status="failed", complete=False, error=repr(exc))
        raise
    finally:
        stop(child)
        stop(server)
        state["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--check-admission", action="store_true")
    args = parser.parse_args()
    if args.check_admission:
        admission(args.task)
        print("admitted")
    else:
        main(args.task)
