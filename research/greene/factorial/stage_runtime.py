"""Create the approved independent runtime from verified frozen predecessor inputs.

Remote staging only. Read a JSON payload on stdin; do not launch any process.
"""

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    payload = json.load(sys.stdin)
    old = Path("/home/zifanz4/openwam-runtime/rae-policy-expanded-20261006")
    root = Path("/home/zifanz4/openwam-runtime/rae-factorial-20261010")
    expected = payload["predecessor_hashes"]
    for file, digest in expected.items():
        if sha(Path(file)) != digest:
            raise RuntimeError(f"frozen predecessor changed: {file}")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir()
    p = json.loads((old / "protocol.json").read_text())
    p.update(
        created=dt.datetime.now(dt.timezone.utc).isoformat(),
        authorization={
            "message": "User replied ok to GPU1/2, idle-only, maximum 36 hours.",
            "budget_start_utc": "2026-10-10T23:52:56+00:00",
            "max_wall_hours": 36,
        },
        deadline="2026-10-12T11:52:56+00:00",
        source_code="/home/zifanz4/openwam-rae-research",
        variants=["raw768_native_shift16", "raw768_wide_shift16"],
        evaluation_variants=[
            "svae48_native",
            "raw768_native",
            "raw768_wide",
            "raw768_native_shift16",
            "raw768_wide_shift16",
        ],
        predecessor_runtime=str(old),
        collection=str(old),
        automatic_retries=False,
        scope="Matched wide-head x video-shift factorial; six new 20000-step runs, "
        "reuse nine frozen shift5 controls. Endpoint is fresh paired RoboTwin success.",
    )
    p["training"]["objective"] = (
        "Unmodified native joint flow loss, lambda_video=lambda_action=1; video shift16.4317, action shift5."
    )
    p["pause_paths"] += [str(root / "paused.json")]
    (root / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    for name in ["cache.pt", "normalization_stats.npy", "dataset-audit.json"]:
        (root / name).symlink_to(old / name)
    (root / "initialization").mkdir()
    for file in (old / "initialization").glob("*.safetensors"):
        (root / "initialization" / file.name).symlink_to(file)
    shutil.copytree(old / "templates", root / "templates")
    for new, base in [("raw768_native_shift16", "raw768_native"), ("raw768_wide_shift16", "raw768_wide")]:
        shutil.copytree(old / "templates" / base, root / "templates" / new)
        (root / "templates" / new / "config.yaml").write_text(payload["configs"][new])
    shutil.copy2(old / "common.py", root / "common.py")
    text = (old / "train.py").read_text()
    before = "if a.variant == 'raw768_wide':"
    assert text.count(before) == 1
    text = text.replace(before, "if getattr(model.video_backbone.dit, 'rae_head', None) is not None:")
    anchor = (
        "    write(run/'result.json',dict(complete=True,variant=a.variant,seed=a.seed,steps=steps,gradients=last_grad,"
    )
    assert text.count(anchor) == 1
    smoke = """    from eval_sigma_curve_v2 import evaluate_curve
    model.eval()
    first_dev = next(i for i, m in enumerate(cache['metadata']) if m['split'] == 'dev')
    diagnostic_cache = dict(cache, metadata=[dict(m, split='dev' if i == first_dev else 'train')
                                           for i, m in enumerate(cache['metadata'])])
    diagnostic = evaluate_curve(model, diagnostic_cache, a.variant, inputs_for, paired_loss,
                                targets=[0.1, 0.5, 0.9], draws=1, guard=guard)
    write(run/'sigma-metric-smoke.json', dict(passed=True, scored_test_scenes_used=False, curve=diagnostic))
"""
    text = text.replace(anchor, smoke + anchor)
    compile(text, str(root / "train.py"), "exec")
    (root / "train.py").write_text(text)
    for name, text in payload["scripts"].items():
        assert Path(name).name == name
        compile(text, name, "exec")
        (root / name).write_text(text)
    # Metadata and immutable inputs are hashed before allowing any GPU worker.
    inputs = {
        str(old / name): sha(old / name) for name in ["cache.pt", "normalization_stats.npy", "dataset-audit.json"]
    }
    inputs.update({str(f): sha(f) for f in (old / "initialization").glob("*.safetensors")})
    (root / "input-freeze.json").write_text(json.dumps(inputs, indent=2) + "\n")
    files = [root / "protocol.json", root / "input-freeze.json", *root.glob("*.py")]
    files += [f for f in (root / "templates").rglob("*") if f.is_file()]
    files += [Path(f) for f in expected if "/openwam-rae-research/" in f]
    freeze = {str(f): sha(f) for f in files}
    (root / "code-freeze.json").write_text(
        json.dumps({"files": freeze, "predecessor": expected, "public_push": False}, indent=2) + "\n"
    )
    (root / "preparation.json").write_text(
        json.dumps(
            {
                "status": "training_ready_pending_gpu",
                "gpu_work_started": False,
                "closed_loop_ready": False,
                "deadline": p["deadline"],
                "input_hashes": inputs,
                "script_config_hashes": freeze,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "runtime": str(root),
                "status": "training_ready_pending_gpu",
                "deadline": p["deadline"],
                "frozen_files": len(freeze),
                "input_files": len(inputs),
            }
        )
    )


if __name__ == "__main__":
    main()
