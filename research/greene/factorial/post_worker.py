"""After training, collect fresh scenes and evaluate every prespecified cell."""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from worker import admit, attempt_limit, gpu_status, now, resource_end, resource_policy, write

ROOT = Path(__file__).resolve().parent


def main(gpu):
    p = json.loads((ROOT / "protocol.json").read_text())
    policy = resource_policy(ROOT, p)
    deadline = dt.datetime.fromisoformat(p["deadline"])
    claims = ROOT / "post-claims"
    claims.mkdir(exist_ok=True)
    statefile = ROOT / f"post-worker-gpu{gpu}.json"
    jobs = [{"stage": "collect", "task": t, "name": "collect-" + t} for t in p["tasks"]]
    jobs += [
        {"stage": "evaluate", "variant": v, "seed": s, "name": f"evaluate-{v}-seed{s}"}
        for s in p["seeds"]
        for v in p["evaluation_variants"]
    ]

    def state(phase, **extra):
        write(statefile, dict(phase=phase, pid=os.getpid(), time=now().isoformat(), gpu=gpu, **extra))

    def guard():
        assert now() < dt.datetime.fromisoformat(p["deadline"]), "deadline reached"
        assert not any(Path(f).exists() for f in p["pause_paths"]), "pause requested"

    def record(job):
        file = claims / (job["name"] + ".json")
        if not file.exists():
            return None
        try:
            return json.loads(file.read_text())
        except json.JSONDecodeError:
            return {"status": "claim_being_written"}

    def terminated(*args):
        raise RuntimeError("termination requested")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    child = None
    streak = 0
    try:
        while True:
            guard()
            if resource_end(policy, gpu, deadline, now()) is None:
                state("waiting_for_authorized_window")
                streak = 0
                time.sleep(30)
                continue
            training = [ROOT / "claims" / f"train-{v}-seed{s}.json" for s in p["seeds"] for v in p["variants"]]
            rows = []
            for file in training:
                try:
                    rows.append(json.loads(file.read_text()))
                except (FileNotFoundError, json.JSONDecodeError):
                    rows.append({})
            if any(r.get("status") == "failed" for r in rows):
                raise RuntimeError("training failed; factorial incomplete")
            if not all(r.get("status") == "complete" for r in rows):
                state("waiting_for_training")
                time.sleep(30)
                continue
            collected = [record(j) for j in jobs[:3]]
            if any(r and r.get("status") == "failed" for r in collected):
                raise RuntimeError("fresh scene preparation failed; no policy scoring")
            eligible = jobs if all(r and r.get("status") == "complete" for r in collected) else jobs[:3]
            job = next((j for j in eligible if record(j) is None), None)
            if job is None:
                allrows = [record(j) for j in jobs]
                if all(r and r.get("status") in ("complete", "failed") for r in allrows):
                    state("finished", records=allrows)
                    return
                state("waiting_for_peer")
                time.sleep(30)
                continue
            if admit(policy, gpu, deadline, now(), job["stage"]) is None:
                state("waiting_for_window_capacity", next_job=job)
                streak = 0
                time.sleep(30)
                continue
            free, values = gpu_status(gpu, policy["gpus"][gpu])
            streak = streak + 1 if free else 0
            state("waiting_for_free_gpu", free_streak=streak, query=values, next_job=job)
            if streak < p["free_polls_required"]:
                time.sleep(p["poll_seconds"])
                continue
            with (ROOT.parent / f"gpu-{gpu}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    streak = 0
                    time.sleep(30)
                    continue
                if not gpu_status(gpu, policy["gpus"][gpu])[0]:
                    streak = 0
                    continue
                guard()
                for file, want in json.loads((ROOT / "eval-freeze.json").read_text())["files"].items():
                    assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == want, file
                attempt_deadline = admit(policy, gpu, deadline, now(), job["stage"])
                if attempt_deadline is None:
                    streak = 0
                    continue
                claim = claims / (job["name"] + ".json")
                row = dict(job, pid=os.getpid(), gpu=gpu, status="running", started=now().isoformat())
                try:
                    with claim.open("x") as handle:
                        json.dump(row, handle)
                except FileExistsError:
                    continue
                runtime = ROOT.parent
                collect = job["stage"] == "collect"
                venv = "benchmark-env" if collect else "policy-env"
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=policy["gpus"][gpu],
                    STUDY_GPU=gpu,
                    EXPECTED_RENDER_PCI=values[4],
                    OMP_NUM_THREADS="4",
                    OPENBLAS_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",
                    CUDA_HOME=str(runtime / "cuda-toolkit"),
                    CPATH=str(runtime / "cuda-toolkit/targets/x86_64-linux/include"),
                    TORCH_CUDA_ARCH_LIST="12.0",
                    TMPDIR=str(runtime / "tmp"),
                    MAX_JOBS="2",
                    TORCH_EXTENSIONS_DIR=str(runtime / "cache/torch_extensions-sm120"),
                    TRITON_CACHE_DIR=str(runtime / "cache/triton-sm120"),
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    TOKENIZERS_PARALLELISM="false",
                    WANDB_MODE="disabled",
                    ROBOTWIN_PATH=str(runtime / "assets-source/benchmarks/RoboTwin"),
                    ROBOTWIN_RUNTIME_ROOT=str(ROOT / f"simulator-runtime-gpu{gpu}"),
                    ROBOTWIN_ENABLE_PLANNER_FALLBACK="1",
                    WAM_ROOT=str(runtime / "assets-source"),
                )
                env["PATH"] = str(runtime / venv / "bin") + ":" + str(runtime / "cuda-toolkit/bin") + ":/usr/bin:/bin"
                env["LD_LIBRARY_PATH"] = str(runtime / "benchmark-env/lib/python3.10/site-packages/sapien/oidn_library")
                if collect:
                    env.pop("PYTHONPATH", None)
                else:
                    env["PYTHONPATH"] = p["source_code"]
                cmd = [
                    str(runtime / venv / "bin/python"),
                    "-u",
                    str(ROOT / ("collect_fresh.py" if collect else "evaluate_cell.py")),
                ]
                cmd += ["--task", job["task"]] if collect else ["--variant", job["variant"], "--seed", str(job["seed"])]
                limit = attempt_limit(job["stage"])
                start = time.monotonic()
                try:
                    with (ROOT / (job["name"] + ".log")).open("x") as log:
                        child = subprocess.Popen(
                            cmd,
                            cwd=p["source_code"],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    row["child_pid"] = child.pid
                    write(claim, row)
                    while child.poll() is None:
                        guard()
                        if now() >= attempt_deadline:
                            raise RuntimeError("resource window cleanup deadline")
                        assert time.monotonic() - start < limit, "attempt limit"
                        state("running", job=row, elapsed_seconds=time.monotonic() - start)
                        time.sleep(5)
                    row.update(status="complete" if child.returncode == 0 else "failed", exit_code=child.returncode)
                except BaseException as exc:
                    row.update(status="failed", error=repr(exc))
                    raise
                finally:
                    if child is not None and child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    child = None
                    row["finished"] = now().isoformat()
                    write(claim, row)
                if not collect:
                    subprocess.run([str(runtime / "policy-env/bin/python"), str(ROOT / "score.py")], check=True)
                streak = 0
    except BaseException as exc:
        state("stopped", error=repr(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=[str(i) for i in range(8)], required=True)
    main(parser.parse_args().gpu)
