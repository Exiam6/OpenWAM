"""Render perturbation vs scene-to-scene distance, measured at two resolutions.

The saved observation PNGs are 240x320 (raw camera). The training latents are on
a 24x20 grid, i.e. 384x320. _preprocess_image does no resizing, so the real
serving path must resize somewhere this script does not reach. Measuring at both
resolutions shows whether the conclusion depends on that step.
"""
import itertools
import json
import pathlib
import sys

import torch
from PIL import Image

study = pathlib.Path(sys.argv[1]).resolve()
variant, out_path = sys.argv[2], sys.argv[3]
sys.path.insert(0, str(study))
from common import build  # noqa: E402

SCENES = pathlib.Path("/home/zifanz4/openwam-runtime/assets-source/temporal-20260926/scenes")
TARGET = SCENES / "place_object_basket/seed-1102008/initial-head_camera.png"
ACT = sorted(pathlib.Path("/home/zifanz4/openwam-runtime/rae-policy-20261001/evaluation").glob(
    "*/place_object_basket/clean/seed-1102008/initial-actual-head_camera.png"))[0]

model, _ = build(variant, 42)
model.set_dtype_device(torch.bfloat16, torch.device("cuda"))
model.eval()
enc = model.video_backbone.video_encoder

others = []
for task in ["place_object_basket", "handover_block", "adjust_bottle"]:
    for d in sorted((SCENES / task).glob("seed-*"))[:6]:
        p = d / "initial-head_camera.png"
        if p.exists() and p != TARGET:
            others.append((task, d.name, p))


def run(resize):
    def encode(path):
        img = Image.open(path).convert("RGB")
        if resize is not None:
            img = img.resize(resize, Image.BICUBIC)
        with torch.no_grad():
            return enc.batch_encode(enc.preprocess_video([img] * 9)).float().cpu().flatten()

    zt = encode(TARGET)
    za = encode(ACT)
    delta = float((za - zt).norm())
    zs = {(t, n): encode(p) for t, n, p in others}
    to_target = sorted(float((z - zt).norm()) for z in zs.values())
    same_task = sorted(float((zs[a] - zs[b]).norm())
                       for a, b in itertools.combinations(zs, 2) if a[0] == b[0])

    def q(v, f):
        return v[int(f * (len(v) - 1))]

    return dict(
        resize=list(resize) if resize else "native 240x320",
        latent_dim=int(zt.numel()),
        latent_norm=float(zt.norm()),
        render_delta=delta,
        render_pct_of_norm=100 * delta / float(zt.norm()),
        nearest_other_scene=to_target[0],
        median_other_scene=q(to_target, 0.5),
        min_same_task_pair=same_task[0],
        median_same_task_pair=q(same_task, 0.5),
        render_vs_nearest_other_pct=100 * delta / to_target[0],
        render_vs_median_other_pct=100 * delta / q(to_target, 0.5),
        render_vs_min_same_task_pct=100 * delta / same_task[0],
    )


results = []
for resize in [None, (320, 384)]:
    r = run(resize)
    results.append(r)
    print(f"\n=== {r['resize']} (latent dim {r['latent_dim']}) ===")
    print(f"  render |dz|            {r['render_delta']:.2f}   ({r['render_pct_of_norm']:.2f}% of |z|)")
    print(f"  nearest other scene    {r['nearest_other_scene']:.2f}")
    print(f"  median other scene     {r['median_other_scene']:.2f}")
    print(f"  min same-task pair     {r['min_same_task_pair']:.2f}")
    print(f"  median same-task pair  {r['median_same_task_pair']:.2f}")
    print(f"  render / nearest other      {r['render_vs_nearest_other_pct']:.1f}%")
    print(f"  render / median other       {r['render_vs_median_other_pct']:.1f}%")
    print(f"  render / min same-task pair {r['render_vs_min_same_task_pct']:.1f}%")

pathlib.Path(out_path).write_text(json.dumps(dict(variant=variant, measurements=results), indent=2) + "\n")
print("\nwrote", out_path)
