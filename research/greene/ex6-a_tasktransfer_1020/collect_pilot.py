"""Freeze expert-feasible scenes before observing any policy outcome."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image

from evaluate_policy import bootstrap_robotwin_module, write
from migration_render import pin_renderer, configure_robot_paths
from collect_scene import pci


def main(task, code):
    protocol = json.loads((code / "pilot-protocol.json").read_text())
    spec = protocol["tasks"][task]
    root = code / "scenes" / task
    root.mkdir(parents=True, exist_ok=False)
    pin_renderer()
    module = bootstrap_robotwin_module()
    configure_robot_paths()
    import openwam2robotwin_interface as interface
    interface._STEP_LIM_OVERRIDES = {}
    template = json.loads((code / "collection-template.json").read_text())
    accepted, attempts = [], []
    def timeout(*_):
        raise TimeoutError("Expert candidate exceeded 180 seconds")
    signal.signal(signal.SIGALRM, timeout)
    for seed in range(spec["seed_start"], spec["seed_start"] + 15):
        dest = root / f"seed-{seed}"
        dest.mkdir()
        args = dict(template, task_name=task, save_path=str(dest), need_plan=True,
                    save_data=False, collect_data=False, eval_mode=True, is_test=True,
                    render_freq=0, eval_video_save_dir=None)
        env = module.class_decorator(task)
        row = {"seed": seed, "status": "started"}
        start = time.monotonic()
        write(dest / "attempt.json", row)
        signal.alarm(180)
        try:
            env.setup_demo(now_ep_num=0, seed=seed, **args)
            assert pci(env.scene.render_system.device.pci_string) == pci(os.environ["EXPECTED_RENDER_PCI"])
            obs = env.get_obs()
            state = interface._extract_eef_proprio(obs).tolist()
            images = {key: np.asarray(obs["observation"][key]["rgb"]).copy()
                      for key in ["head_camera", "left_camera", "right_camera"]}
            poses = {}
            for name, actor in vars(env).items():
                if hasattr(actor, "get_pose"):
                    pose = actor.get_pose()
                    poses[name] = {"p": np.asarray(pose.p).tolist(), "q": np.asarray(pose.q).tolist()}
            env.play_once()
            feasible = bool(env.plan_success and env.check_success())
            row.update(status="complete", expert_success=feasible)
            if feasible:
                write(dest / "collection-config.json", args)
                write(dest / "initial-replay.json", {"state": state, "poses": poses})
                for key, image in images.items():
                    Image.fromarray(image).save(dest / f"initial-{key}.png")
                accepted.append({"seed": seed, "instruction": spec["instruction"]})
        except Exception as exc:
            row.update(status="expert_failure", error=repr(exc))
        finally:
            signal.alarm(0)
            try:
                env.close_env(clear_cache=True)
            except Exception as exc:
                row["cleanup_error"] = repr(exc)
                row["status"] = "expert_failure"
                accepted = [r for r in accepted if r["seed"] != seed]
            row["seconds"] = time.monotonic() - start
            write(dest / "attempt.json", row)
            attempts.append(row)
            write(root / "manifest.json", {"complete": len(accepted) == 5,
                  "accepted": accepted, "attempts": attempts,
                  "rule": "First five expert-feasible scenes within fifteen fixed candidates; no policy screening"})
        if len(accepted) == 5:
            break
    assert len(accepted) == 5, "Insufficient feasible scenes; retain failures without replacement"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    main(args.task, args.code)
