"""Dev video loss as a function of noise level, on a sigma grid shared across arms.

The arms carry different shift values, so their scheduler sigma buffers differ and
compute_loss's own sampling would evaluate them at different noise. Pinning the
timestep boundaries to a single index (min=k/N, max=(k+1)/N makes randint(k, k+1)
return k) lets each arm be evaluated at whichever of ITS buffer entries sits
closest to a shared target sigma, so the comparison is at matched corruption.

Usage: eval_sigma_curve.py <study_root> <variant> <checkpoint> <out.json>
"""
import json
import pathlib
import sys

import torch

study = pathlib.Path(sys.argv[1]).resolve()
variant, ckpt_path, out_path = sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, str(study))

from common import P, activate, build, inputs_for, paired_loss  # noqa: E402

TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

cache = torch.load(study / "cache.pt", map_location="cpu", weights_only=False)
dev_ids = [i for i, m in enumerate(cache["metadata"]) if m["split"] == "dev"]
print(f"{len(dev_ids)} dev windows", flush=True)

model, _ = build(variant, 42)
activate(model)
model.load_checkpoint(ckpt_path)
model.eval()

sigmas = model.video_backbone.scheduler.sigmas.float().cpu()
n = len(model.video_backbone.scheduler.timesteps)
print(f"sigma buffer: n={n} min={sigmas.min():.4f} max={sigmas.max():.4f}", flush=True)

rows = []
with torch.no_grad():
    for target in TARGETS:
        k = int(torch.argmin((sigmas[:n] - target).abs()).item())
        actual = float(sigmas[k])
        # Pin the draw to index k for every window.
        lo, hi = k / n, (k + 1) / n
        vals = []
        for j, i in enumerate(dev_ids):
            inp = inputs_for(cache, [i], variant, checkpoint=False)
            inp["min_timestep_boundary"] = lo
            inp["max_timestep_boundary"] = hi
            out = paired_loss(model, inp, 7000000 + i * 10)
            vals.append(float(out["loss_video"]))
        mean = sum(vals) / len(vals)
        rows.append(dict(target_sigma=target, actual_sigma=actual, index=k, loss_video=mean, n=len(vals)))
        print(f"  target {target:.2f} -> sigma {actual:.4f} (idx {k})  loss_video {mean:.6f}", flush=True)

result = dict(study=str(study), variant=variant, checkpoint=ckpt_path,
              shift_video=float(getattr(model.video_backbone, "shift_video", float("nan"))),
              dev_windows=len(dev_ids), curve=rows)
pathlib.Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
print("wrote", out_path)
