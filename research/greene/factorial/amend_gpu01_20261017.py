"""Move the paused RAE factorial study to shenlong-gpu-01 under the 2026-10-17 authorization.

Run on gpu-01 after the runtime has been mirrored to the same absolute paths.
Scientific protocol fields (data, recipe, arms, seeds, scenes, endpoint) are not
touched; only deadline, GPU map, polling, scheduler safety limits and the idle
rule change. Nothing is launched here.
"""

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

R = Path("/home/zifanz4/openwam-runtime/rae-factorial-20261010")
DEADLINE = "2026-10-18T17:00:00+00:00"  # 2026-10-18 12:00 America/Chicago
USER_MESSAGE = "明天中午12点前 这些卡全都可以用 充分利用 最大化gpu 自主推进 利用率低的也可以用 / gpu01类似的也可以"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert subprocess.check_output(["hostname"], text=True).strip().startswith("shenlong-gpu-01")
    p = json.loads((R / "protocol.json").read_text())
    archive = R / "resource-amendment-20261017-gpu01"
    assert not archive.exists(), "amendment already applied"
    # 1. Every frozen file must have arrived intact before anything is changed.
    stale = []
    for name in ["code-freeze.json", "eval-freeze.json"]:
        for path, want in json.loads((R / name).read_text())["files"].items():
            if path.endswith("resource-policy.json"):
                continue
            if not Path(path).exists() or sha(path) != want:
                stale.append(path)
    for path, want in json.loads((R / "input-freeze.json").read_text()).items():
        if sha(path) != want:
            stale.append(path)
    assert not stale, stale
    for pause in p["pause_paths"]:
        assert not Path(pause).exists(), pause
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    gpus = dict(tuple(x.strip() for x in line.split(",")) for line in lines.splitlines())
    assert set(gpus) == set(map(str, range(8)))
    # 2. Archive what is replaced; gpu-02 preflight evidence is kept there too.
    archive.mkdir()
    for name in ["protocol.json", "resource-policy.json", "code-freeze.json", "eval-freeze.json", "worker.py",
                 "post_worker.py", "monitor.py", "worker-services.json", "post-worker-services.json",
                 "evaluate_policy.py", "score.py"]:
        shutil.copy2(R / name, archive / name)
    (archive / "gpu02-preflight").mkdir()
    for claim in (R / "claims").glob("preflight-*.json"):
        shutil.move(str(claim), archive / "gpu02-preflight" / claim.name)
    for log in R.glob("preflight-*.log"):
        shutil.move(str(log), archive / "gpu02-preflight" / log.name)
    if (R / "preflight").exists():
        shutil.move(str(R / "preflight"), archive / "gpu02-preflight" / "preflight")
    assert not list((R / "claims").glob("*.json")) and not list((R / "post-claims").glob("*.json"))
    # 3. Authorization, deadline, GPU map, polling.
    p["gpus"] = gpus
    p["deadline"] = DEADLINE
    p["free_polls_required"] = 3
    p["authorization"] = {
        "message": USER_MESSAGE,
        "granted_at_local": "2026-10-17 ~01:50 America/Chicago",
        "host": "shenlong-gpu-01",
        "all_eight_gpus": True,
        "shared_low_utilization_gpus_allowed": True,
        "deadline": DEADLINE,
        "previous": json.loads((archive / "protocol.json").read_text())["authorization"],
    }
    p["execution_host_note"] = (
        "Training uses precomputed latents (no rendering). Scene screening and all 900 closed-loop attempts "
        "run on this one host so every arm shares one renderer/driver; gpu-02 results are not mixed in."
    )
    (R / "protocol.json").write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n")
    policy = dict(
        host="shenlong-gpu-01", timezone="America/Chicago", start_hour=0, end_hour=24,
        baseline_gpus=sorted(gpus), gpus=gpus, idle_only=False, allow_shared=True,
        shared_max_utilization_percent=30, shared_min_free_mib=40000,
        user_message=USER_MESSAGE, recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        study_deadline=DEADLINE, max_attempt_cleanup_margin_seconds=120, launcher="setsid-nohup",
    )
    (R / "resource-policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n")
    # 4. Scheduler edits (worker.py): shared-GPU rule and a longer training time limit.
    w = (R / "worker.py").read_text()
    old_q = '"--query-gpu=uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total,pci.bus_id",'
    new_q = '"--query-gpu=uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total,pci.bus_id,memory.total",'
    assert w.count(old_q) == 1
    w = w.replace(old_q, new_q)
    old_ret = "    return int(values[1]) < 100 and int(values[2]) == 0 and not apps, values\n"
    new_ret = (
        "    idle = int(values[1]) < 100 and int(values[2]) == 0 and not apps\n"
        "    policy = json.loads((ROOT / 'resource-policy.json').read_text())\n"
        "    shared = (\n"
        "        bool(policy.get('allow_shared'))\n"
        "        and int(values[2]) <= int(policy['shared_max_utilization_percent'])\n"
        "        and int(values[5]) - int(values[1]) >= int(policy['shared_min_free_mib'])\n"
        "    )\n"
        "    values.append('idle' if idle else ('shared' if shared else 'busy'))\n"
        "    return idle or shared, values\n"
    )
    assert w.count(old_ret) == 1
    w = w.replace(old_ret, new_ret)
    old_lim = '{"preflight": 1800, "train": 10800, "collect": 7200, "evaluate": 14400}'
    new_lim = '{"preflight": 1800, "train": 14400, "collect": 7200, "evaluate": 14400}'
    assert w.count(old_lim) == 1
    w = w.replace(old_lim, new_lim)
    compile(w, "worker.py", "exec")
    (R / "worker.py").write_text(w)
    # 5. monitor.py: observe setsid processes instead of systemd units.
    m = (R / "monitor.py").read_text()
    old_obs = '''            observed = command(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "-p",
                    "ActiveState",
                    "-p",
                    "SubState",
                    "-p",
                    "MainPID",
                    "-p",
                    "Result",
                ]
            )
            status = {}
            if observed.get("returncode") == 0:
                status = dict(line.split("=", 1) for line in observed["stdout"].splitlines())
            state = read(ROOT / f"{prefix}worker-gpu{gpu}.json")
'''
    new_obs = '''            state = read(ROOT / f"{prefix}worker-gpu{gpu}.json")
            pid = state.get("pid")
            alive = False
            if pid:
                try:
                    import os
                    os.kill(int(pid), 0)
                    alive = Path(f"/proc/{pid}/cmdline").read_bytes().find(b"worker.py") >= 0
                except (OSError, ValueError):
                    alive = False
            observed = {"returncode": 0, "stdout": "", "stderr": ""}
            status = {"ActiveState": "active" if alive else ("failed" if state.get("error") else "inactive"),
                      "SubState": state.get("phase", ""), "MainPID": str(pid or 0), "Result": ""}
'''
    assert m.count(old_obs) == 1
    m = m.replace(old_obs, new_obs)
    old_stop = '        result = command(["systemctl", "--user", "stop", TIMER])\n'
    new_stop = '        result = {"note": "no timer on gpu-01; the session loop stops polling"}\n'
    assert m.count(old_stop) == 1
    m = m.replace(old_stop, new_stop)
    old_budget = '    if not claims["claims"] and remaining < 28:'
    new_budget = '    if not claims["claims"] and remaining < 12:'
    assert m.count(old_budget) == 1
    m = m.replace(old_budget, new_budget)
    compile(m, "monitor.py", "exec")
    (R / "monitor.py").write_text(m)
    # 5b. evaluate_policy.py: keep every policy-input frame (JPEG per policy call) so the
    #     report page can show what the policy saw. Pure logging; success rule untouched.
    e = (R / "evaluate_policy.py").read_text()
    old_tick = "      tick_before=execution['physics_ticks'];call_before=len(execution['planner_calls'])\n"
    new_tick = (
        "      frames=dest/'frames';frames.mkdir(exist_ok=True)\n"
        "      for key in ['head_camera','left_camera','right_camera']:Image.fromarray(np.asarray(obs['observation'][key]['rgb'])).save(frames/f\"{current['step']:04d}-{key}.jpg\",quality=85)\n"
        + old_tick
    )
    assert e.count(old_tick) == 1
    e = e.replace(old_tick, new_tick)
    old_break = "      if env.eval_success:break\n"
    new_break = (
        old_break
        + "     final=env.get_obs()\n"
        + "     for key in ['head_camera','left_camera','right_camera']:Image.fromarray(np.asarray(final['observation'][key]['rgb'])).save(dest/'frames'/f\"{int(env.take_action_cnt):04d}-{key}.jpg\",quality=85)\n"
    )
    assert e.count(old_break) == 1
    e = e.replace(old_break, new_break)
    compile(e, "evaluate_policy.py", "exec")
    (R / "evaluate_policy.py").write_text(e)
    # 5c. score.py: single-variable paired contrasts (rule 12); file comes from the research branch.
    new_score = Path(__file__).resolve().parent / "score.py"
    (R / "score.py").write_text(new_score.read_text())
    # 6. Refresh manifests for exactly the changed files.
    changed = ["protocol.json", "resource-policy.json", "worker.py", "monitor.py", "evaluate_policy.py", "score.py"]
    for name in ["code-freeze.json", "eval-freeze.json"]:
        manifest = json.loads((R / name).read_text())
        for c in changed:
            manifest["files"][str(R / c)] = sha(R / c)
        manifest["resource_amendment_gpu01"] = str(archive)
        (R / name).write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ["worker-services.json", "post-worker-services.json"]:
        (R / name).write_text("[]\n")
    record = dict(
        applied_at=dt.datetime.now(dt.timezone.utc).isoformat(), host="shenlong-gpu-01", policy=policy,
        deadline=DEADLINE, changed_files={c: sha(R / c) for c in changed},
        archived=str(archive), gpu02_preflight_moved=True,
        scientific_fields_unchanged=["tasks", "variants", "evaluation_variants", "seeds", "steps", "training",
                                     "head", "data", "episodes", "evaluation", "automatic_retries"],
    )
    (R / "resource-amendment-gpu01.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "policy"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
