"""Does the render divergence change the policy's ACTION, not just its latent?

Feeds the production serving path two observations that differ only in which
render they carry - the recorded reference or the replayed one - with identical
proprio, identical instruction, and the same _study_action_seed so the server
resets and seeds identically before each. Any difference in the returned action
is attributable to the render alone.
"""
import json
import pathlib
import statistics
import sys
import time

import numpy as np
from PIL import Image

E = pathlib.Path("/home/zifanz4/openwam-runtime/assets-source/temporal-20260926")
STUDY = pathlib.Path("/home/zifanz4/openwam-runtime/rae-policy-20261001")
SURVEY = pathlib.Path("/home/zifanz4/openwam-runtime/render-survey-20261003")
port, out_path = int(sys.argv[1]), sys.argv[2]

sys.path.insert(0, str(E / "OpenWAM/benchmarks/robotwin"))
import openwam2robotwin_interface as interface  # noqa: E402

model = interface.ModelClient(host="127.0.0.1", port=port, send_state=True,
                              state_dim=20, request_timeout=120, action_type="ee")
current = {"seed": 0}
raw_predict = model._client.predict_once


def send(payload):
    payload = dict(payload)
    # step 0 makes the server reset and seed before predicting, so both members
    # of a pair start from exactly the same RNG state.
    payload.update(_study_step=0, _study_action_seed=current["seed"])
    return raw_predict(payload)


model._client.predict = send

INSTR = json.loads((STUDY / "eval-protocol.json").read_text())["instructions"]
CAMS = ["head_camera", "left_camera", "right_camera"]
survey = json.loads((SURVEY / "survey.json").read_text())
ok = [r for r in survey["records"] if r["status"] == "ok"]
print(f"{len(ok)} scenes", flush=True)


def images(task, seed, which):
    out = {}
    for cam, key in zip(CAMS, ["head", "left", "right"]):
        p = (E / "scenes" / task / f"seed-{seed}" / f"initial-{cam}.png") if which == "ref" \
            else (SURVEY / task / f"seed-{seed}" / f"actual-{cam}.png")
        out[key] = np.asarray(Image.open(p).convert("RGB"))
    return out


rows = []
begin = time.monotonic()
for r in ok:
    task, seed = r["task"], r["scene_seed"]
    state = np.asarray(json.loads((E / "scenes" / task / f"seed-{seed}" / "initial-replay.json")
                                  .read_text())["state"], dtype=np.float32)
    current["seed"] = int(np.random.SeedSequence([seed, 42, 811]).generate_state(1, dtype=np.uint32)[0])
    acts = {}
    for which in ["ref", "actual"]:
        interface.reset_model(model)
        acts[which] = np.asarray(model.step(dict(
            cams=images(task, seed, which), lang=INSTR[task], state=state)), dtype=np.float32)
    d = acts["actual"] - acts["ref"]
    scale = float(np.abs(acts["ref"]).max())
    rows.append(dict(task=task, scene_seed=seed, image_mae=r["mae"]["head_camera"],
                     action_dim=int(acts["ref"].size),
                     l2=float(np.linalg.norm(d)), linf=float(np.abs(d).max()),
                     ref_l2=float(np.linalg.norm(acts["ref"])),
                     relative_l2=float(np.linalg.norm(d) / max(np.linalg.norm(acts["ref"]), 1e-9)),
                     ref_absmax=scale))
    print(f"  {task:<22} seed{seed}  |da|={rows[-1]['l2']:.5f}  "
          f"rel={100*rows[-1]['relative_l2']:.2f}%  linf={rows[-1]['linf']:.5f}", flush=True)
    pathlib.Path(out_path).write_text(json.dumps(dict(rows=rows), indent=2) + "\n")

rel = sorted(r["relative_l2"] for r in rows)
l2 = sorted(r["l2"] for r in rows)
summary = dict(
    scenes=len(rows), seconds=time.monotonic() - begin,
    relative_l2=dict(min=rel[0], median=statistics.median(rel), max=rel[-1]),
    absolute_l2=dict(min=l2[0], median=statistics.median(l2), max=l2[-1]),
    identical=sum(1 for r in rows if r["l2"] == 0.0),
    note="Same proprio, same instruction, same action seed; only the render differs.",
)
pathlib.Path(out_path).write_text(json.dumps(dict(summary=summary, rows=rows), indent=2) + "\n")
print()
print(f"relative action change: min {100*rel[0]:.3f}%  median {100*statistics.median(rel):.3f}%  max {100*rel[-1]:.3f}%")
print(f"absolute |da|         : min {l2[0]:.5f}  median {statistics.median(l2):.5f}  max {l2[-1]:.5f}")
print(f"bit-identical actions : {summary['identical']}/{len(rows)}")
print("wrote", out_path)
