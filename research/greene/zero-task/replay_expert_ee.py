"""ex4a: replay expert end-effector states through the policy action path (eef20d_to_ee16d -> take_action 'ee').

Usage (benchmark-env python, cwd = RoboTwin runtime set by the wrapper):
  replay_expert_ee.py <output_dir> <demo_dir> [<demo_dir> ...]
No learned policy is loaded. Writes <output_dir>/<demo id>.json per demonstration and summary.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

STUDY = Path("/home/zifanz4/openwam-runtime/rae-factorial-20261010")
sys.path.insert(0, str(STUDY))
from migration_render import configure_robot_paths, pin_renderer  # noqa: E402

pin_renderer()
sys.path.insert(0, "/home/zifanz4/openwam-runtime/assets-source/temporal-20260926/OpenWAM/benchmarks/robotwin")
from eval_policy_wrapper import bootstrap_robotwin_module  # noqa: E402

mod = bootstrap_robotwin_module()
configure_robot_paths()
import openwam2robotwin_interface as interface  # noqa: E402
import yaml  # noqa: E402
from execution_trace import track_execution  # noqa: E402
from sapien.wrapper.scene import Scene  # noqa: E402

conv = interface.action_conversion


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    args = yaml.safe_load(Path("task_config/demo_clean.yml").read_text())
    emb = yaml.safe_load(Path("task_config/_embodiment_config.yml").read_text())["aloha-agilex"]["file_path"]
    ec = mod.get_embodiment_config(emb)
    camera = mod.get_camera_config(args["camera"]["head_camera_type"])
    args.update(task_config="demo_clean", embodiment_name="aloha-agilex", left_robot_file=emb, right_robot_file=emb,
                left_embodiment_config=ec, right_embodiment_config=ec, dual_arm_embodied=True, head_camera_h=camera["h"],
                head_camera_w=camera["w"], need_plan=True, save_data=False, render_freq=0, eval_mode=True, is_test=True,
                eval_video_save_dir=None)
    rows = []
    for d in sys.argv[2:]:
        d = Path(d)
        st = json.loads((d / "status.json").read_text())
        traj = np.load(d / "full-trajectory.npz")
        prop = traj["proprio"]
        task, seed = st["task"], int(st["seed"])
        args["task_name"] = task
        args["save_path"] = str(out / f"{task}-{seed}")
        env = mod.class_decorator(task)
        t0 = time.monotonic()
        row = dict(id=st["id"], task=task, seed=seed, demo_frames=int(len(prop)))
        try:
            env.setup_demo(now_ep_num=0, seed=seed, **args)
            obs = env.get_obs()
            init = np.asarray(interface._extract_eef_proprio(obs))
            row["initial_proprio_max_abs_diff"] = float(np.abs(init - prop[0]).max())
            limit = int(env.step_lim)
            row["step_limit"] = limit
            planner = []
            with track_execution(env, Scene) as ex:
                for t in range(1, len(prop)):
                    before = len(ex["planner_calls"])
                    env.take_action(conv.eef20d_to_ee16d(np.asarray(prop[t], dtype=np.float64)), action_type="ee")
                    calls = ex["planner_calls"][before:]
                    planner.append({c["side"]: c.get("status", "EXC") for c in calls})
                    if env.eval_success:
                        break
            reached = np.asarray(interface._extract_eef_proprio(env.get_obs()))
            row.update(success=bool(env.eval_success), check_success=bool(env.check_success()), steps=int(env.take_action_cnt),
                       physics_ticks=int(ex["physics_ticks"]),
                       left_fail=sum(1 for p in planner if p.get("left") not in (None, "Success")),
                       right_fail=sum(1 for p in planner if p.get("right") not in (None, "Success")),
                       first_fail_step=next((i + 1 for i, p in enumerate(planner) if any(v != "Success" for v in p.values())), None),
                       final_pos_err_left=float(np.linalg.norm(reached[0:3] - prop[min(len(prop) - 1, env.take_action_cnt)][0:3])),
                       final_pos_err_right=float(np.linalg.norm(reached[10:13] - prop[min(len(prop) - 1, env.take_action_cnt)][10:13])))
        except Exception as exc:  # report, never hide
            row.update(error=repr(exc))
        finally:
            env.close_env(clear_cache=True)
        row["seconds"] = round(time.monotonic() - t0, 1)
        (out / f"{row['id']}.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "summary.json").write_text(json.dumps(dict(host=os.uname().nodename, rows=rows), indent=2))


if __name__ == "__main__":
    main()
