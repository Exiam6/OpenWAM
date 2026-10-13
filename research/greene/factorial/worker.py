"""Bounded per-GPU queue for the authorized RAE factorial study.

A claimed attempt is never reclaimed automatically. All children share a process
session so pause/deadline cleanup covers the training process and descendants.
"""

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
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent


def write(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc)


def resource_policy(root, protocol):
    path = root / "resource-policy.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"gpus": protocol["gpus"], "baseline_gpus": list(protocol["gpus"])}


def resource_end(policy, gpu, deadline, stamp):
    """Return the UTC end of the currently authorized interval, or None."""
    if gpu not in policy["gpus"] or stamp >= deadline:
        return None
    if gpu in policy["baseline_gpus"]:
        return deadline
    local = stamp.astimezone(ZoneInfo(policy["timezone"]))
    start, end = policy["start_hour"], policy["end_hour"]
    if not start <= local.hour < end:
        return None
    end = local.replace(hour=end, minute=0, second=0, microsecond=0, fold=0)
    return min(deadline, end.astimezone(dt.timezone.utc))


def attempt_limit(stage):
    return {"preflight": 1800, "train": 10800, "collect": 7200, "evaluate": 14400}[stage]


def admit(policy, gpu, deadline, stamp, stage):
    end = resource_end(policy, gpu, deadline, stamp)
    if end is None:
        return None
    if gpu not in policy["baseline_gpus"]:
        # Full worst-case runtime plus startup/poll/cleanup margin must fit.
        if (end - stamp).total_seconds() < attempt_limit(stage) + 120:
            return None
        return end - dt.timedelta(seconds=60)
    return end


def gpu_status(gpu, expected):
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-gpu=uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    )
    values = [x.strip() for x in line.strip().split(",")]
    if values[0] != expected:
        raise RuntimeError("GPU UUID mismatch")
    apps = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20
    ).strip()
    if values[3] != "0":
        raise RuntimeError("GPU reports uncorrected ECC errors")
    return int(values[1]) < 100 and int(values[2]) == 0 and not apps, values


def command(job, python):
    args = [str(python), "-u", str(ROOT / "train.py"), job["variant"], str(job["seed"])]
    if job["stage"] == "preflight":
        args.append("--preflight")
    return args


def run(gpu):
    p = json.loads((ROOT / "protocol.json").read_text())
    deadline = dt.datetime.fromisoformat(p["deadline"])
    policy = resource_policy(ROOT, p)
    uuid = policy["gpus"][gpu]
    runtime = ROOT.parent
    jobs = [{"stage": "preflight", "variant": v, "seed": 42} for v in p["variants"]]
    jobs += [{"stage": "train", "variant": v, "seed": s} for s in p["seeds"] for v in p["variants"]]
    claims = ROOT / "claims"
    claims.mkdir(exist_ok=True)
    statefile = ROOT / f"worker-gpu{gpu}.json"
    child = None

    def guard():
        if now() >= deadline:
            raise RuntimeError("authorization deadline reached")
        for path in p["pause_paths"]:
            if Path(path).exists():
                raise RuntimeError(f"pause requested: {path}")

    def state(phase, **extra):
        write(statefile, dict(time=now().isoformat(), pid=os.getpid(), gpu=gpu, phase=phase, **extra))

    def stopped(*unused):
        raise RuntimeError("worker termination requested")

    def result(job):
        name = f"{job['stage']}-{job['variant']}-seed{job['seed']}"
        path = claims / f"{name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {"status": "claim_being_written"}

    def available():
        preflights = jobs[: len(p["variants"])]
        records = [result(job) for job in preflights]
        if any(row and row.get("status") == "failed" for row in records):
            raise RuntimeError("preflight failed; production disabled")
        if any(not row or row.get("status") != "complete" for row in records):
            return next((job for job in preflights if result(job) is None), None)
        return next((job for job in jobs[len(preflights) :] if result(job) is None), None)

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    streak = 0
    try:
        while True:
            guard()
            if resource_end(policy, gpu, deadline, now()) is None:
                state("waiting_for_authorized_window")
                streak = 0
                time.sleep(30)
                continue
            job = available()
            if job is None:
                records = [result(x) for x in jobs]
                if all(row and row.get("status") in ("complete", "failed") for row in records):
                    state(
                        "training_queue_finished",
                        completed=sum(r["status"] == "complete" for r in records),
                        failed=sum(r["status"] == "failed" for r in records),
                        evaluation="not_started",
                    )
                    return
                state("waiting_for_peer", records=records)
                time.sleep(30)
                continue
            if admit(policy, gpu, deadline, now(), job["stage"]) is None:
                state("waiting_for_window_capacity", next_job=job)
                streak = 0
                time.sleep(30)
                continue
            free, values = gpu_status(gpu, uuid)
            streak = streak + 1 if free else 0
            state("waiting_for_free_gpu", free_streak=streak, query=values, next_job=job)
            if streak < p["free_polls_required"]:
                time.sleep(p["poll_seconds"])
                continue
            lock = (runtime / f"gpu-{gpu}.lock").open("a")
            try:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    streak = 0
                    time.sleep(30)
                    continue
                free, values = gpu_status(gpu, uuid)
                if not free:
                    streak = 0
                    continue
                guard()
                # Check every frozen code/config file again before each attempt.
                manifest = json.loads((ROOT / "code-freeze.json").read_text())["files"]
                for path, expected in manifest.items():
                    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                        raise RuntimeError(f"frozen source changed: {path}")
                job = available()
                if job is None:
                    continue
                attempt_deadline = admit(policy, gpu, deadline, now(), job["stage"])
                if attempt_deadline is None:
                    streak = 0
                    continue
                name = f"{job['stage']}-{job['variant']}-seed{job['seed']}"
                claim = claims / f"{name}.json"
                row = dict(job, status="running", pid=os.getpid(), gpu=gpu, started=now().isoformat())
                try:
                    with claim.open("x") as handle:
                        json.dump(row, handle)
                except FileExistsError:
                    continue
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=uuid,
                    STUDY_GPU=gpu,
                    PYTHONPATH=p["source_code"],
                    OMP_NUM_THREADS="4",
                    OPENBLAS_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",
                    MAX_JOBS="2",
                    CUDA_HOME=str(runtime / "cuda-toolkit"),
                    TORCH_CUDA_ARCH_LIST="12.0",
                    TMPDIR=str(runtime / "tmp"),
                    TORCH_EXTENSIONS_DIR=str(runtime / "cache/torch_extensions-sm120"),
                    TRITON_CACHE_DIR=str(runtime / "cache/triton-sm120"),
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    TOKENIZERS_PARALLELISM="false",
                    WANDB_MODE="disabled",
                )
                env["PATH"] = (
                    str(runtime / "policy-env/bin") + ":" + str(runtime / "cuda-toolkit/bin") + ":/usr/bin:/bin"
                )
                with (ROOT / f"{name}.log").open("x") as log:
                    child = subprocess.Popen(
                        command(job, runtime / "policy-env/bin/python"),
                        cwd=p["source_code"],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                row["child_pid"] = child.pid
                write(claim, row)
                state("running", job=row)
                start = time.monotonic()
                limit = attempt_limit(job["stage"])
                try:
                    while child.poll() is None:
                        guard()
                        if now() >= attempt_deadline:
                            raise RuntimeError("resource window cleanup deadline")
                        if time.monotonic() - start > limit:
                            raise RuntimeError("attempt time limit")
                        state("running", job=row, elapsed_seconds=time.monotonic() - start)
                        time.sleep(5)
                    resultpath = (
                        ROOT
                        / ("preflight" if job["stage"] == "preflight" else "runs")
                        / f"{job['variant']}-seed{job['seed']}"
                        / "result.json"
                    )
                    ok = (
                        child.returncode == 0
                        and resultpath.exists()
                        and json.loads(resultpath.read_text()).get("complete")
                    )
                    row.update(status="complete" if ok else "failed", exit_code=child.returncode)
                except BaseException as exc:
                    row.update(status="failed", error=repr(exc))
                    raise
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    row["finished"] = now().isoformat()
                    write(claim, row)
                    child = None
                streak = 0
            finally:
                lock.close()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state("stopped", error=repr(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, choices=[str(i) for i in range(8)])
    run(parser.parse_args().gpu)
