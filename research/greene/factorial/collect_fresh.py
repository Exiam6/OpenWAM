"""Freeze the first 20 expert-feasible scenes in 40 prespecified candidates.

No learned policy is loaded. Every attempted candidate is retained. Infrastructure
failures stop the task; expert infeasibility is the only exclusion criterion.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent / "assets-source/temporal-20260926"


def write(path, record):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)


def capture(env, dest, interface):
    obs = env.get_obs()
    poses = {}
    for name, value in vars(env).items():
        if name.startswith("_") or not callable(getattr(value, "get_pose", None)):
            continue
        try:
            pose = value.get_pose()
            poses[name] = {"p": np.asarray(pose.p).tolist(), "q": np.asarray(pose.q).tolist()}
        except (AttributeError, TypeError):
            pass
    state = np.asarray(interface._extract_eef_proprio(obs), dtype=np.float32)
    for camera in ["head_camera", "left_camera", "right_camera"]:
        Image.fromarray(obs["observation"][camera]["rgb"]).save(dest / f"initial-{camera}.png")
    record = {"state": state.tolist(), "poses": poses, "eval_mode": bool(env.eval_mode)}
    write(dest / "initial-replay.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    p = json.loads((ROOT / "protocol.json").read_text())
    ep = json.loads((ROOT / "scene-protocol.json").read_text())

    def guard():
        assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(p["deadline"])
        assert not any(Path(x).exists() for x in p["pause_paths"])

    def timeout(*unused):
        raise TimeoutError("expert candidate exceeded 300 seconds")

    signal.signal(signal.SIGALRM, timeout)
    from migration_render import configure_robot_paths, pin_renderer

    pin_renderer()
    sys.path.insert(0, str(ASSETS / "OpenWAM/benchmarks/robotwin"))
    from eval_policy_wrapper import bootstrap_robotwin_module

    mod = bootstrap_robotwin_module()
    configure_robot_paths()
    import openwam2robotwin_interface as interface
    import yaml
    from collect_scene import pci

    task = args.task
    assert task in ep["seed_starts"]
    folder = ROOT / "scenes" / task
    folder.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(Path("task_config/demo_clean.yml").read_text())
    emb = yaml.safe_load(Path("task_config/_embodiment_config.yml").read_text())["aloha-agilex"]["file_path"]
    ec = mod.get_embodiment_config(emb)
    camera = mod.get_camera_config(config["camera"]["head_camera_type"])
    config.update(
        task_name=task,
        task_config="demo_clean",
        embodiment_name="aloha-agilex",
        left_robot_file=emb,
        right_robot_file=emb,
        left_embodiment_config=ec,
        right_embodiment_config=ec,
        dual_arm_embodied=True,
        head_camera_h=camera["h"],
        head_camera_w=camera["w"],
        need_plan=True,
        save_data=False,
        render_freq=0,
        eval_mode=True,
        is_test=True,
        eval_video_save_dir=None,
    )
    accepted = []
    attempts = []
    for offset in range(ep["max_candidates_per_task"]):
        guard()
        seed = ep["seed_starts"][task] + offset
        dest = folder / f"seed-{seed}"
        dest.mkdir()
        config["save_path"] = str(dest)
        write(dest / "collection-config.json", config)
        row = {"task": task, "seed": seed, "accepted": False, "instruction": ep["instructions"][task]}
        env = None
        started = time.monotonic()
        signal.alarm(300)
        try:
            env = mod.class_decorator(task)
            env.setup_demo(now_ep_num=0, seed=seed, **config)
            assert pci(env.scene.render_system.device.pci_string) == pci(os.environ["EXPECTED_RENDER_PCI"])
            reference = capture(env, dest, interface)
            env.play_once()
            feasible = bool(env.plan_success and env.check_success())
            row.update(accepted=feasible, outcome="accepted" if feasible else "expert_infeasible")
            if feasible:
                env.close_env(clear_cache=True)
                env = None
                env = mod.class_decorator(task)
                env.setup_demo(now_ep_num=0, seed=seed, **config)
                obs = env.get_obs()
                assert np.allclose(interface._extract_eef_proprio(obs), reference["state"], rtol=0, atol=1e-6)
                for name, pose in reference["poses"].items():
                    actual = getattr(env, name).get_pose()
                    assert np.allclose(actual.p, pose["p"], rtol=0, atol=1e-6)
                    assert np.allclose(actual.q, pose["q"], rtol=0, atol=1e-6)
                errors = {
                    camera: float(
                        np.abs(
                            np.asarray(obs["observation"][camera]["rgb"], dtype=float)
                            - np.asarray(Image.open(dest / f"initial-{camera}.png"), dtype=float)
                        ).mean()
                    )
                    for camera in ["head_camera", "left_camera", "right_camera"]
                }
                assert max(errors.values()) <= 1.0, errors
                row["same_renderer_replay_mae"] = errors
        except BaseException as exc:
            row.update(accepted=False, outcome="infrastructure_failure", error=repr(exc))
            raise
        finally:
            signal.alarm(0)
            if env is not None:
                env.close_env(clear_cache=True)
            row["seconds"] = time.monotonic() - started
            write(dest / "result.json", row)
            attempts.append(row)
            if row["accepted"]:
                accepted.append(row)
            write(folder / "progress.json", {"accepted": accepted, "attempts": attempts})
        if len(accepted) == ep["accepted_per_task"]:
            break
    result = {
        "task": task,
        "accepted": accepted,
        "attempts": attempts,
        "complete": len(accepted) == ep["accepted_per_task"],
        "no_learned_policy_scoring": True,
    }
    write(folder / "manifest.json", result)
    assert result["complete"], "fixed candidate range exhausted; no replacement range"
    hashes = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in folder.rglob("*")
        if f.is_file() and f.suffix in (".json", ".png")
    }
    write(folder / "freeze.json", hashes)


if __name__ == "__main__":
    main()
