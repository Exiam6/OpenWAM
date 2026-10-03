"""Latent-space render sensitivity across all 60 scenes.

Extends the single-scene result. For each scene the survey saved the replayed
render next to its recorded reference, so every pair can be encoded and compared
in one latent space, against the scene-to-scene distances measured in that same
space.
"""
import itertools
import json
import pathlib
import statistics
import sys

import torch
from PIL import Image

study = pathlib.Path(sys.argv[1]).resolve()
variant, survey_dir, out_path = sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(study))
from common import build  # noqa: E402

SCENES = pathlib.Path("/home/zifanz4/openwam-runtime/assets-source/temporal-20260926/scenes")
CAM = "head_camera"
RESIZE = (320, 384)  # the training latent grid; the native 240x320 agreed to within 1.3%

model, _ = build(variant, 42)
model.set_dtype_device(torch.bfloat16, torch.device("cuda"))
model.eval()
enc = model.video_backbone.video_encoder


def encode(path):
    img = Image.open(path).convert("RGB").resize(RESIZE, Image.BICUBIC)
    with torch.no_grad():
        return enc.batch_encode(enc.preprocess_video([img] * 9)).float().cpu().flatten()


survey = json.loads((survey_dir / "survey.json").read_text())
ok = [r for r in survey["records"] if r["status"] == "ok"]
print(f"{len(ok)} scenes surveyed", flush=True)

refs, rows = {}, []
for r in ok:
    task, seed = r["task"], r["scene_seed"]
    ref_p = SCENES / task / f"seed-{seed}" / f"initial-{CAM}.png"
    act_p = survey_dir / task / f"seed-{seed}" / f"actual-{CAM}.png"
    if not act_p.exists():
        print(f"  missing render for {task} seed{seed}", flush=True)
        continue
    z_ref = encode(ref_p)
    z_act = encode(act_p)
    refs[(task, seed)] = z_ref
    rows.append(dict(task=task, scene_seed=seed, image_mae=r["mae"][CAM],
                     delta=float((z_act - z_ref).norm()),
                     pct_of_norm=100 * float((z_act - z_ref).norm()) / float(z_ref.norm())))
    print(f"  {task:<22} seed{seed}  mae={r['mae'][CAM]:.4f}  |dz|={rows[-1]['delta']:.2f}"
          f"  {rows[-1]['pct_of_norm']:.2f}% of |z|", flush=True)

# Scene-to-scene distances in the same space.
same_task = sorted(float((refs[a] - refs[b]).norm())
                   for a, b in itertools.combinations(refs, 2) if a[0] == b[0])
cross = sorted(float((refs[a] - refs[b]).norm())
               for a, b in itertools.combinations(refs, 2) if a[0] != b[0])

deltas = sorted(r["delta"] for r in rows)
med_delta = statistics.median(deltas)


def q(v, f):
    return v[int(f * (len(v) - 1))]


summary = dict(
    scenes=len(rows), camera=CAM, resize=list(RESIZE),
    render_delta=dict(min=deltas[0], median=med_delta, max=deltas[-1],
                      pct_of_norm_median=statistics.median(r["pct_of_norm"] for r in rows)),
    image_mae=dict(min=min(r["image_mae"] for r in rows),
                   median=statistics.median(r["image_mae"] for r in rows),
                   max=max(r["image_mae"] for r in rows)),
    same_task_pairs=dict(n=len(same_task), min=same_task[0], median=q(same_task, 0.5), max=same_task[-1]),
    cross_task_pairs=dict(n=len(cross), min=cross[0], median=q(cross, 0.5), max=cross[-1]),
    render_vs_min_same_task_pct=100 * med_delta / same_task[0],
    render_vs_median_same_task_pct=100 * med_delta / q(same_task, 0.5),
    scenes_whose_render_exceeds_min_same_task_pair=sum(1 for d in deltas if d > same_task[0]),
)
pathlib.Path(out_path).write_text(json.dumps(dict(summary=summary, rows=rows), indent=2) + "\n")

print()
print(f"render |dz|       : min {deltas[0]:.2f}  median {med_delta:.2f}  max {deltas[-1]:.2f}")
print(f"  as % of |z|     : median {summary['render_delta']['pct_of_norm_median']:.2f}%")
print(f"same-task pairs   : min {same_task[0]:.2f}  median {q(same_task,0.5):.2f}  (n={len(same_task)})")
print(f"cross-task pairs  : min {cross[0]:.2f}  median {q(cross,0.5):.2f}  (n={len(cross)})")
print()
print(f"median render / min same-task pair    : {summary['render_vs_min_same_task_pct']:.1f}%")
print(f"median render / median same-task pair : {summary['render_vs_median_same_task_pct']:.1f}%")
print(f"scenes whose render delta exceeds the closest same-task scene pair: "
      f"{summary['scenes_whose_render_exceeds_min_same_task_pair']}/{len(rows)}")
print("wrote", out_path)
