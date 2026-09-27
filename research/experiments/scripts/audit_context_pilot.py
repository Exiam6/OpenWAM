"""Post-run audit of every context record and frozen input; no inference."""

import argparse
import base64
import gzip
import io
import json
from pathlib import Path

import numpy as np
from context_runner import file_hash
from PIL import Image
from repeat_common import array_hash, payload_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--study", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    study = a.study
    freeze = json.loads((study / "source-manifest.json").read_text())
    for name, h in freeze["runtime_files"].items():
        assert file_hash(root / name) == h, name
    for name, h in freeze["study_files"].items():
        assert file_hash(study / name) == h, name
    summary = json.loads((study / "summary.json").read_text())
    requests = images = planners = 0
    videos = []
    for name, t in summary["trials"].items():
        d = study / name
        for row in t["records"]:
            payload = json.loads(
                gzip.decompress((d / row["request_file"]).read_bytes())
            )
            assert payload_hash(payload) == row["request_sha256"]
            requests += 1
            assert np.array_equal(
                np.asarray(payload["state"], np.float32),
                np.asarray(row["eef_state"], np.float32),
            )
            for src, key in [
                ("head_camera", "head_camera"),
                ("left_camera", "left_wrist_camera"),
                ("right_camera", "right_wrist_camera"),
            ]:
                im = np.array(
                    Image.open(io.BytesIO(base64.b64decode(payload["images"][key])))
                )
                assert array_hash(im) == row["raw_cameras"][src]
                images += 1
                if row["step"] == 1 and src == "head_camera":
                    Image.fromarray(im).save(d / "step1-head.png")
            assert [c["side"] for c in row["planner_calls"]] == ["left", "right"]
            planners += 2
            assert (
                row["requests"] == row["actions"] == 1
                and row["after_action_count"] == row["step"] + 1
            )
        for video in d.glob("runtime/**/*.mp4"):
            videos.append(
                {
                    "trial": name,
                    "source": str(video.relative_to(study)),
                    "sha256": file_hash(video),
                    "bytes": video.stat().st_size,
                }
            )
    audit = {
        "all_completed_record_checks_passed": True,
        "runtime_sources_verified": len(freeze["runtime_files"]),
        "study_inputs_verified": len(freeze["study_files"]),
        "requests_audited": requests,
        "decoded_images_audited": images,
        "planner_calls_recorded": planners,
        "complete_trials": summary["completed_trials"],
        "planned_trials": 4,
        "videos": videos,
        "partial_or_failed_trials_retained": True,
    }
    (study / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
