"""Does per-scene render divergence predict failure? GPU-free, existing data only.

adjust_bottle is the one task where the official checkpoint under OIDN 2.3.3
has outcome variance (9/20). The survey measured each scene's initial-frame
render MAE under the same OIDN 2.3.3, and action_sensitivity measured each
scene's step-0 action deviation (for raw768_native, so a weaker proxy).
If divergence drives the failures, failed scenes should sit higher on both.
Not the intervention, but it is evidence we can have now.
"""
import json
import pathlib
import random

import numpy as np

R = pathlib.Path("/home/zifanz4/openwam-runtime")
outcome = {}
for f in (R / "official-shenlong/adjust_bottle").glob("seed-*/result.json"):
    r = json.loads(f.read_text())
    if r.get("status") == "complete":
        outcome[r["scene_seed"]] = bool(r["success"])

survey = {r["scene_seed"]: r for r in json.loads((R / "render-survey-20261003/survey.json").read_text())["records"]
          if r["task"] == "adjust_bottle" and r["status"] == "ok"}
action = {r["scene_seed"]: r for r in json.loads((R / "action-sens/result.json").read_text())["rows"]
          if r["task"] == "adjust_bottle"}

scenes = sorted(set(outcome) & set(survey))
print(f"adjust_bottle scenes with both outcome and survey: {len(scenes)}  "
      f"(success {sum(outcome[s] for s in scenes)}, fail {sum(not outcome[s] for s in scenes)})")


def mann_whitney(a, b, n_perm=200000, seed=0):
    """Rank-biserial effect and a permutation p-value for 'b tends to exceed a'."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    def u_stat(x, y):
        return sum((yy > xx) + 0.5 * (yy == xx) for xx in x for yy in y)
    u = u_stat(a, b)
    rb = 2 * u / (len(a) * len(b)) - 1  # +1: every fail > every success
    pooled = np.concatenate([a, b]); na = len(a)
    rng = random.Random(seed); hits = 0
    for _ in range(n_perm):
        idx = list(range(len(pooled))); rng.shuffle(idx)
        pa = pooled[idx[:na]]; pb = pooled[idx[na:]]
        if u_stat(pa, pb) >= u: hits += 1
    return rb, hits / n_perm


for label, key, src in [
    ("initial-frame render MAE, max over 3 cameras (same OIDN 2.3.3 as the eval)", "max_mae", survey),
    ("initial-frame render MAE, head camera only", None, survey),
    ("step-0 action deviation, raw768_native-seed42 (proxy model)", "relative_l2", action),
]:
    def val(s):
        if key is None: return src[s]["mae"]["head_camera"]
        return src[s][key]
    ok = [val(s) for s in scenes if outcome[s] and s in src]
    bad = [val(s) for s in scenes if not outcome[s] and s in src]
    if not ok or not bad:
        print(f"\n{label}: insufficient coverage"); continue
    rb, p = mann_whitney(ok, bad)
    print(f"\n{label}")
    print(f"  success (n={len(ok):2d}): median {np.median(ok):.4f}  range {min(ok):.4f}–{max(ok):.4f}")
    print(f"  fail    (n={len(bad):2d}): median {np.median(bad):.4f}  range {min(bad):.4f}–{max(bad):.4f}")
    print(f"  rank-biserial (fail > success) = {rb:+.3f}   one-sided permutation p = {p:.4f}")

print("\nper-scene:")
print(f"{'scene':>9} {'ok':>3} {'max_mae':>8} {'head_mae':>9} {'act_dev%':>9}")
for s in scenes:
    a = action.get(s, {}).get("relative_l2")
    print(f"{s:>9} {'Y' if outcome[s] else '-':>3} {survey[s]['max_mae']:8.4f} {survey[s]['mae']['head_camera']:9.4f} "
          f"{(100*a if a is not None else float('nan')):9.2f}")
