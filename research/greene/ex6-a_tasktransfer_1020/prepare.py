"""Create a separate, immutable pilot; register only after CPU validation."""
import datetime as dt
import json
from pathlib import Path
import shutil

from run_task import ROOT, RUNTIME, SOURCE, sha

TASKS = {
    "click_bell": "Click the bell.",
    "press_stapler": "Press the stapler.",
    "open_laptop": "Open the laptop.",
    "pick_dual_bottles": "Pick up both bottles.",
    "lift_pot": "Lift the pot.",
    "stack_blocks_two": "Stack the two blocks.",
    "place_bread_basket": "Place the bread into the basket.",
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    ROOT.mkdir(exist_ok=False)
    template = ROOT / "template"
    template.mkdir()
    old = RUNTIME / "ex5-a_vaebaseline_1018/evaluation-template"
    for name in ["evaluate_policy.py", "serve_policy.py", "simulation_smoke.py",
                 "migration_render.py", "collect_scene.py", "execution_trace.py"]:
        shutil.copy2(old / name, template / name)
    path = template / "evaluate_policy.py"
    text = path.read_text()
    assert text.count("len(manifest['accepted'])==20") == 1
    text = text.replace("len(manifest['accepted'])==20", "len(manifest['accepted'])==5")
    # Native RoboTwin first executes the expert, then resets the SAME task
    # instance. Some tasks initialize success-rule attributes in play_once
    # (e.g. open_laptop.arm_tag); skipping that step breaks their evaluator.
    setup = "    env.setup_demo(now_ep_num=0,seed=seed,**args)\n"
    assert text.count(setup) == 1
    text = text.replace(setup, setup +
        "    env.play_once()\n"
        "    assert env.plan_success and env.check_success(),'Frozen expert scene no longer feasible'\n"
        "    env.close_env(clear_cache=True)\n" + setup)
    path.write_text(text)
    shutil.copy2(SOURCE / "collect_pilot.py", template / "collect_pilot.py")
    shutil.copy2(SOURCE / "run_task.py", ROOT / "run_task.py")
    shutil.copy2(SOURCE / "prepare.py", ROOT / "prepare.py")
    shutil.copy2(RUNTIME / "ex5-a_vaebaseline_1018/base-environment.json", ROOT / "base-environment.json")
    shutil.copytree(RUNTIME / "ex5-a_vaebaseline_1018/robot-configs", ROOT / "robot-configs",
                    symlinks=True)
    shutil.copy2(RUNTIME / "rae-factorial-20261010/scenes/adjust_bottle/seed-6200000/collection-config.json",
                 template / "collection-template.json")
    audit_path = RUNTIME / "rae-policy-expanded-20261006/dataset-audit.json"
    audit = json.loads(audit_path.read_text())
    identities = list(audit["source_hashes"])
    training_tasks = sorted({identity.split("-seed")[0] for identity in identities})
    training_seeds = {int(identity.split("-seed")[1]) for identity in identities}
    tasks = {name: {"instruction": prompt, "seed_start": 7300000 + 1000*i}
             for i, (name, prompt) in enumerate(TASKS.items())}
    candidates = {v["seed_start"] + i for v in tasks.values() for i in range(15)}
    assert candidates.isdisjoint(training_seeds)
    assert set(tasks).isdisjoint(training_tasks)
    dump(template / "seed-exclusion.json", {"passed": True, "audit": str(audit_path),
         "audit_sha256": sha(audit_path), "training_tasks": training_tasks,
         "new_tasks": list(tasks), "candidate_seeds": sorted(candidates),
         "scope": "Small-model training corpus only; official checkpoint is a task-trained reference"})
    previous = json.loads((old / "eval-protocol.json").read_text())
    previous["instructions"].update(TASKS)
    previous.update(scope="Seven new-task transfer pilots; original tasks only for unscored transport smoke",
                    accepted_per_task=5, max_candidates_per_task=15, budget="7 tasks x 5 scenes x 4 arms")
    dump(template / "eval-protocol.json", previous)
    dump(template / "scene-protocol.json", previous)
    dump(template / "pilot-protocol.json", {"tasks": tasks})
    arms = []
    for route in ["raw768_native", "svae48_native"]:
        result = json.loads((RUNTIME / "rae-factorial-20261010/runs" / f"{route}-seed42/result.json").read_text())
        assert result["complete"] and result["steps"] == 20000
        arms.append({"name": route, "checkpoint": result["checkpoint"], "sha256": result["checkpoint_sha256"]})
    result = json.loads((RUNTIME / "ex5-a_vaebaseline_1018/runs/vae48_native-seed42/result.json").read_text())
    assert result["complete"] and result["steps"] == 20000
    arms.append({"name": "vae48_native", "checkpoint": result["checkpoint"], "sha256": result["checkpoint_sha256"]})
    official = json.loads((RUNTIME / "ex4-h_officialcalibration_1017/registration.json").read_text())
    # The official model is a functional reference, not a matched representation arm.
    arms.insert(0, {"name": "official_reference", "checkpoint": official["checkpoint"],
                    "sha256": official["checkpoint_sha256"]})
    code_paths = [p for p in ROOT.rglob("*") if p.is_file()]
    dependency_paths = [audit_path]
    simulator = RUNTIME / "assets-source/benchmarks/RoboTwin"
    for base in [simulator / "envs", simulator / "task_config",
                 RUNTIME / "ex5-a_vaebaseline_1018/corrected-engine-v1/openwam"]:
        dependency_paths.extend(p for p in base.rglob("*") if p.suffix in [".py", ".yaml", ".yml"])
    dependency_paths.append(RUNTIME / "assets-source/temporal-20260926/OpenWAM/benchmarks/robotwin/eval_policy_wrapper.py")
    for arm in arms:
        dependency_paths.extend(Path(arm["checkpoint"]).parent / name for name in ["config.yaml", "normalization_stats.npy"])
    now = dt.datetime.now(dt.timezone.utc)
    p = {
        "name": ROOT.name, "created": now.isoformat(),
        "deadline": (now + dt.timedelta(days=3)).isoformat(),
        "user_instruction": "再多测试一些任务 openwam官方都做了哪些wval",
        "resource_instruction": "01:00–09:30 America/Chicago; 30min observed idle; prefer gpu01 GPUs1/2",
        "tasks": tasks, "arms": arms, "training_seeds": [42],
        "interpretation": "Exploratory unseen-task transfer for three small models trained only on the original three tasks. Official checkpoint is a differently pretrained, larger functional reference. Not a matched official leaderboard reproduction or a multi-seed RAE superiority test.",
        "selection": "Tasks fixed by interaction diversity before outcomes. First 5 expert-feasible candidates from 15 fixed seeds. Never select tasks or scenes by policy success.",
        "job_max_seconds": 14400, "total_gpu_hour_cap": 28,
        "planning_estimate_gpu_hours": [7, 14],
        "estimate_basis": "140 scored episodes at roughly 2–4 minutes plus expert checks, loading and transport smoke. New tasks may be slower; 600s episode cap and 4h task cap retain unfinished attempts.",
        "engine": str(RUNTIME / "ex5-a_vaebaseline_1018/corrected-engine-v1"),
        "evaluation": {"condition": "demo_clean", "episodes_per_task_arm": 5, "planned_episodes": 140,
                       "native_step_limits": True, "episode_seconds": 600,
                       "denoise_steps": 10, "inference_horizon": 8, "compile": False, "dit_cache": False,
                       "metric": "Native task success, per-task counts; technical failures reported separately",
                       "reporting": "Do not exclude failed tasks from an overall success claim; do not infer a population effect from one training seed."},
        "code_sha256": {str(path): sha(path) for path in code_paths + dependency_paths},
        "automatic_retries": False, "training_authorized_by_this_protocol": False,
    }
    dump(ROOT / "protocol.json", p)
    print(json.dumps({"prepared": True, "tasks": list(tasks), "planned_episodes": 140,
                      "gpu_jobs_registered": False, "protocol_sha256": sha(ROOT / "protocol.json")}))


if __name__ == "__main__":
    prepare()
