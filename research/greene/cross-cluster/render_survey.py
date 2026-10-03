"""Re-render every recorded scene's initial state and compare to its reference.

Extends the render-divergence measurement from the single scene that happened to
trip the gate to all 60. No policy, no scoring, no success judgment, so the
attempt-once discipline does not apply - this only sets up each scene and reads
the first observation.
"""
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image

E = pathlib.Path("/home/zifanz4/openwam-runtime/assets-source/temporal-20260926")
STUDY = pathlib.Path("/home/zifanz4/openwam-runtime/rae-policy-20261001")
OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(E / "OpenWAM/benchmarks/robotwin"))

from migration_render import configure_robot_paths, pin_renderer  # noqa: E402
from eval_policy_wrapper import bootstrap_robotwin_module  # noqa: E402

pin_renderer()
mod = bootstrap_robotwin_module()
configure_robot_paths()

CAMERAS = ["head_camera", "left_camera", "right_camera"]
TASKS = ["adjust_bottle", "handover_block", "place_object_basket"]

rows = []
begin = time.monotonic()
for task in TASKS:
    manifest = json.loads((E / "scenes" / task / "manifest.json").read_text())
    env = mod.class_decorator(task)
    for scene in manifest["accepted"]:
        seed = int(scene["seed"])
        src = E / "scenes" / task / f"seed-{seed}"
        dest = OUT / task / f"seed-{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        args = json.loads((src / "collection-config.json").read_text())
        args.update(save_path=str(dest), need_plan=True, save_data=False,
                    eval_mode=True, is_test=True, render_freq=0, eval_video_save_dir=None)
        t0 = time.monotonic()
        try:
            env.setup_demo(now_ep_num=0, seed=seed, **args)
            obs = env.get_obs()
            errors = {}
            for cam in CAMERAS:
                actual = np.asarray(obs["observation"][cam]["rgb"])
                reference = np.asarray(Image.open(src / f"initial-{cam}.png"))
                errors[cam] = float(np.abs(actual.astype(float) - reference.astype(float)).mean())
                Image.fromarray(actual).save(dest / f"actual-{cam}.png")
            row = dict(task=task, scene_seed=seed, status="ok", mae=errors,
                       max_mae=max(errors.values()), seconds=time.monotonic() - t0)
        except Exception as exc:  # a scene that cannot be set up is recorded, not retried
            row = dict(task=task, scene_seed=seed, status="failed", error=repr(exc),
                       seconds=time.monotonic() - t0)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (OUT / "survey.json").write_text(json.dumps(
            dict(records=rows, seconds=time.monotonic() - begin), indent=2) + "\n")

ok = [r for r in rows if r["status"] == "ok"]
maxes = sorted(r["max_mae"] for r in ok)
summary = dict(
    scenes=len(rows), ok=len(ok), failed=len(rows) - len(ok),
    max_mae=dict(
        min=maxes[0], median=maxes[len(maxes) // 2], max=maxes[-1],
        over_gate_1_0=sum(1 for v in maxes if v > 1.0),
    ),
    per_task={t: dict(
        n=sum(1 for r in ok if r["task"] == t),
        median=sorted(r["max_mae"] for r in ok if r["task"] == t)[sum(1 for r in ok if r["task"] == t) // 2],
        max=max(r["max_mae"] for r in ok if r["task"] == t),
    ) for t in TASKS if any(r["task"] == t for r in ok)},
    seconds=time.monotonic() - begin,
)
(OUT / "survey.json").write_text(json.dumps(dict(records=rows, summary=summary), indent=2) + "\n")
print(json.dumps(summary, indent=2))
