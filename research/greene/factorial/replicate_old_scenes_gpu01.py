"""Cross-host reproduction control (EXPERIMENT_PROTOCOL rules 23/29): re-run one ex2 cell on gpu-01.

raw768_native seed 42, adjust_bottle, the 20 old scenes, clean condition, same frozen checkpoint and
policy server as ex2, and compare scene by scene with gpu-02's ledger rows. Uses a copy of the ex3
evaluation scripts with only the scene location changed (old scenes instead of fresh ones). Runs on
one GPU that no study worker is using; writes everything under the given output directory.
Usage: replicate_old_scenes_gpu01.py <gpu index> <output dir>
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

RT = Path("/home/zifanz4/openwam-runtime")
S = RT / "rae-factorial-20261010"
OLD = RT / "assets-source/temporal-20260926"
LEDGER = Path("/home/zifanz4/openwam-torch-dispatch/research/greene/records/rae-factorial-preparation-20261010/originals/evaluation-ledger.json")
ROUTE, SEED, TASK = "raw768_native", 42, "adjust_bottle"


def main(gpu, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    work = out / "scripts"
    work.mkdir()
    for name in ["evaluate_policy.py", "serve_policy.py", "simulation_smoke.py", "migration_render.py", "collect_scene.py", "execution_trace.py", "eval-protocol.json"]:
        shutil.copy2(S / name, work / name)
    # Only change: scenes come from the old sealed cohort; the fresh-cohort exclusion check does not apply.
    e = (work / "evaluate_policy.py").read_text()
    e = e.replace("S=Path(__file__).resolve().parent;E=S", f"S=Path(__file__).resolve().parent;E=Path('{OLD}')")
    e = e.replace("E/'scene-protocol.json'", "E/'protocol.json'")
    e = e.replace(" assert json.loads((S/'seed-exclusion.json').read_text())['passed']\n", " # old sealed cohort: fresh-cohort exclusion check not applicable\n")
    assert e.count(str(OLD)) == 1 and "seed-exclusion" not in e
    (work / "evaluate_policy.py").write_text(e)
    uuid, pci = [l.split(", ") for l in subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"], text=True).splitlines()][0]
    port = 19900 + int(gpu)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
    checkpoint = S / "runs" / f"{ROUTE}-seed{SEED}" / "checkpoint_step_20000.safetensors"
    base = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, STUDY_GPU=gpu, EXPECTED_RENDER_PCI=pci.strip(), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                CUDA_HOME=str(RT / "cuda-toolkit"), CPATH=str(RT / "cuda-toolkit/targets/x86_64-linux/include"), TORCH_CUDA_ARCH_LIST="12.0", TMPDIR=str(RT / "tmp"), MAX_JOBS="2",
                TORCH_EXTENSIONS_DIR=str(RT / "cache/torch_extensions-sm120"), TRITON_CACHE_DIR=str(RT / "cache/triton-sm120"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled", ROBOTWIN_PATH=str(RT / "assets-source/benchmarks/RoboTwin"), ROBOTWIN_RUNTIME_ROOT=str(out / "simulator-runtime"),
                ROBOTWIN_ENABLE_PLANNER_FALLBACK="1", WAM_ROOT=str(RT / "assets-source"), LD_LIBRARY_PATH=str(RT / "benchmark-env/lib/python3.10/site-packages/sapien/oidn_library"))
    pol = dict(base, PATH=str(RT / "policy-env/bin") + ":" + str(RT / "cuda-toolkit/bin") + ":/usr/bin:/bin", PYTHONPATH="/home/zifanz4/openwam-rae-research")
    sim = dict(base, PATH=str(RT / "benchmark-env/bin") + ":" + str(RT / "cuda-toolkit/bin") + ":/usr/bin:/bin")
    sim.pop("PYTHONPATH", None)
    cwd = "/home/zifanz4/openwam-rae-research"
    log = (out / "server.log").open("a")
    server = subprocess.Popen([str(RT / "policy-env/bin/python"), "-u", str(work / "serve_policy.py"), "--checkpoint", str(checkpoint), "--port", str(port), "--log", str(out / "episode-rng.jsonl")],
                              env=pol, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t = time.monotonic()
        while "ENDTOEND_SERVER_READY" not in (out / "server.log").read_text(errors="replace"):
            assert server.poll() is None, "policy server failed"
            assert time.monotonic() - t < 900, "policy load timeout"
            time.sleep(3)
        smoke = out / "training-scene-smoke"
        smoke.mkdir()
        rc = subprocess.run([str(RT / "benchmark-env/bin/python"), "-u", str(work / "simulation_smoke.py"), "--port", str(port), "--output", str(smoke)], env=sim, cwd=cwd,
                            stdout=(smoke / "run.log").open("a"), stderr=subprocess.STDOUT).returncode
        assert rc == 0 and json.loads((smoke / "result.json").read_text())["passed"], "simulator gate failed"
        (work / f"simulation-preflight-{ROUTE}-{SEED}.json").write_text(json.dumps(dict(passed=True, checkpoint=str(checkpoint), code_sha256={}, development_result=str(smoke / "result.json"))))
        manifest = json.loads((OLD / "scenes" / TASK / "manifest.json").read_text())
        seeds = [int(r["seed"]) for r in manifest["accepted"]]
        target = out / TASK / "clean"
        rc = subprocess.run([str(RT / "benchmark-env/bin/python"), "-u", str(work / "evaluate_policy.py"), "--scene-seeds", json.dumps(seeds), "--task", TASK, "--route", ROUTE, "--seed", str(SEED),
                             "--condition", "clean", "--port", str(port), "--output", str(target)], env=sim, cwd=cwd, stdout=(out / "evaluate.log").open("a"), stderr=subprocess.STDOUT).returncode
        assert rc == 0, "evaluate_policy failed"
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
    # scene-by-scene comparison with gpu-02
    ledger = {r["scene_seed"]: r for r in json.loads(LEDGER.read_text())["records"] if r["route"] == ROUTE and r["training_seed"] == SEED and r["task"] == TASK}
    rows = []
    for seed in seeds:
        new = json.loads((target / f"seed-{seed}" / "result.json").read_text())
        old = ledger[seed]
        rows.append(dict(scene_seed=seed, gpu02_success=old.get("success"), gpu02_steps=old.get("steps"), gpu01_success=new.get("success"), gpu01_steps=new.get("steps"), gpu01_status=new["status"],
                         initial_state_identical=old["initial_state"] == new.get("initial_state"), action_seed_identical=old.get("action_seed") == new.get("action_seed")))
    agree = sum(r["gpu02_success"] == r["gpu01_success"] for r in rows)
    summary = dict(route=ROUTE, training_seed=SEED, task=TASK, scenes=len(rows), gpu02_successes=sum(bool(r["gpu02_success"]) for r in rows), gpu01_successes=sum(bool(r["gpu01_success"]) for r in rows),
                   scene_agreement=agree, initial_state_identical=all(r["initial_state_identical"] for r in rows), action_seed_identical=all(r["action_seed_identical"] for r in rows), rows=rows,
                   note="Same checkpoint, scenes, server settings and per-episode RNG seeds as ex2 on gpu-02; GPU kernels differ by host so bitwise identity is not expected.")
    (out / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
