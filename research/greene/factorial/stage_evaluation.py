"""Stage evaluation code separately from the immutable running training worker."""

import hashlib
import json
import shutil
import sys
from pathlib import Path


def main():
    payload = json.load(sys.stdin)
    root = Path("/home/zifanz4/openwam-runtime/rae-factorial-20261010")
    old = root.parent / "rae-policy-expanded-20261006"
    p = json.loads((root / "protocol.json").read_text())
    for file, want in json.loads((root / "code-freeze.json").read_text())["files"].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == want, file
    assert not (root / "eval-freeze.json").exists()
    ep = json.loads((old / "eval-protocol.json").read_text())
    ep.update(
        seed_starts=dict(zip(p["tasks"], [6200000, 6201000, 6202000])),
        accepted_per_task=20,
        max_candidates_per_task=40,
        base_port=19800,
        scope="Fresh expert-filtered cohort; first20 feasible within40 fixed candidates per task; no policy selection.",
        evaluation={"conditions": ["clean"]},
    )
    ep.pop("scene_sets")
    ep["budget"] = "5 arms x 3 training seeds x 3 tasks x 20 fresh scenes = 900 planned attempts"
    candidates = {v + i for v in ep["seed_starts"].values() for i in range(40)}
    checked = {}

    def integers(value):
        if isinstance(value, int):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from integers(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from integers(item)

    paths = set()
    for study in root.parent.iterdir():
        if study.is_dir() and study.name.startswith(("rae-", "temporal-", "migration-")) and study != root:
            paths.update(study.glob("*protocol*.json"))
            paths.update(study.rglob("manifest.json"))
    paths.add(root.parent / "assets-source/temporal-20260926/protocol.json")
    paths.update((root.parent / "assets-source/temporal-20260926/scenes").glob("*/manifest.json"))
    for path in sorted(paths):
        data = path.read_bytes()
        try:
            record = json.loads(data)
        except json.JSONDecodeError:
            continue
        overlap = candidates & set(integers(record))
        assert not overlap, (str(path), sorted(overlap))
        checked[str(path)] = hashlib.sha256(data).hexdigest()
    (root / "seed-exclusion.json").write_text(
        json.dumps(
            {
                "passed": True,
                "candidate_seeds": sorted(candidates),
                "checked_known_protocols_and_manifests": checked,
                "training_source": p["predecessor_runtime"],
            },
            indent=2,
        )
        + "\n"
    )
    for name in ["scene-protocol.json", "eval-protocol.json"]:
        (root / name).write_text(json.dumps(ep, indent=2) + "\n")
    for name in ["serve_policy.py", "simulation_smoke.py", "migration_render.py", "collect_scene.py"]:
        shutil.copy2(old / name, root / name)
    for name, source in payload["scripts"].items():
        assert Path(name).name == name
        compile(source, name, "exec")
        (root / name).write_text(source)
    # Reuse native rollout logic, changing only scene location and adding traces.
    source = (old / "evaluate_policy.py").read_text()
    source = source.replace(
        "E=Path('/home/zifanz4/openwam-runtime/assets-source/temporal-20260926');S=Path(__file__).resolve().parent",
        "S=Path(__file__).resolve().parent;E=S",
    )
    source = source.replace(
        "str(E/'OpenWAM/benchmarks/robotwin')",
        "'/home/zifanz4/openwam-runtime/assets-source/temporal-20260926/OpenWAM/benchmarks/robotwin'",
    )
    source = source.replace("E/'protocol.json'", "E/'scene-protocol.json'")
    source = source.replace(
        " assert json.loads((Path('/home/zifanz4/openwam-runtime/temporal-new-20260928/resume-v1')/'fresh-cohort-integrity.json').read_text())['passed']",
        " assert json.loads((S/'seed-exclusion.json').read_text())['passed']",
    )
    anchor = "    with (dest/'trace.jsonl').open('x') as trace:"
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        "    from execution_trace import track_execution\n    from sapien.wrapper.scene import Scene\n    with track_execution(env,Scene) as execution, (dest/'trace.jsonl').open('x') as trace:",
    )
    anchor = "      altered=noise_observation(obs,sigma,seed,current['step']);interface.eval(env,model,altered)"
    assert source.count(anchor) == 1
    source = source.replace(
        anchor, "      tick_before=execution['physics_ticks'];call_before=len(execution['planner_calls'])\n" + anchor
    )
    source = source.replace(
        "'gripper_contacts':contacts}",
        "'gripper_contacts':contacts,'physics_ticks':execution['physics_ticks'],'physics_ticks_this_action':execution['physics_ticks']-tick_before,'physics_timestep':float(env.scene.get_timestep()),'planner_calls':execution['planner_calls'][call_before:]}",
    )
    compile(source, "evaluate_policy.py", "exec")
    (root / "evaluate_policy.py").write_text(source)
    source = (old / "evaluate.py").read_text()
    source = source.replace("import study_queue as Q\n", "")
    source = source[: source.index("\nrows=[]")]
    source = source.replace("ROOT/'code-freeze.json'", "ROOT/'eval-freeze.json'")
    anchor = "EP=json.loads((ROOT/'eval-protocol.json').read_text())"
    source = source.replace(
        anchor,
        anchor
        + "\nEP['scene_sets']={'fresh':{task:[r['seed'] for r in json.loads((ROOT/'scenes'/task/'manifest.json').read_text())['accepted']] for task in P['tasks']}}\nfor task in P['tasks']:\n manifest=json.loads((ROOT/'scenes'/task/'manifest.json').read_text());assert manifest['complete'] and len(manifest['accepted'])==20\n for file,want in json.loads((ROOT/'scenes'/task/'freeze.json').read_text()).items():assert sha(ROOT/file)==want,file",
    )
    source += """

def termination(*unused):
 raise RuntimeError('Evaluation termination requested')
signal.signal(signal.SIGTERM,termination)
signal.signal(signal.SIGINT,termination)
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--variant',required=True,choices=P['evaluation_variants'])
parser.add_argument('--seed',required=True,type=int,choices=P['seeds'])
args=parser.parse_args()
record=run_cell('fresh',args.variant,args.seed)
assert record['status']=='attempts_finished',record
assert all(row['exit_code']==0 for row in record['groups']),record
"""
    compile(source, "evaluate_cell.py", "exec")
    (root / "evaluate_cell.py").write_text(source)
    (root / "runs").mkdir(exist_ok=True)
    for variant in ["svae48_native", "raw768_native", "raw768_wide"]:
        for seed in p["seeds"]:
            (root / "runs" / f"{variant}-seed{seed}").symlink_to(
                old / "runs" / f"{variant}-seed{seed}", target_is_directory=True
            )
    files = dict(json.loads((root / "code-freeze.json").read_text())["files"])
    for f in list(root.glob("*.py")) + [
        root / "scene-protocol.json",
        root / "eval-protocol.json",
        root / "seed-exclusion.json",
    ]:
        files[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    (root / "eval-freeze.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    print(
        json.dumps(
            {
                "evaluation_staged": True,
                "known_manifests_checked": len(checked),
                "candidate_seeds": 120,
                "frozen_files": len(files),
                "deadline": p["deadline"],
            }
        )
    )


if __name__ == "__main__":
    main()
