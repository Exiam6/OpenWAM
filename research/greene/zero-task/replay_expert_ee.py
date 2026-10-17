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
if os.environ.get("REPLAY_PLANNER") == "mplib":
    # Force RoboTwin's planner fallback (mplib) by making cuRobo construction fail; diagnostic only.
    from envs.robot import planner as _planner_mod
    def _no_curobo(self, *a, **k):
        raise RuntimeError("cuRobo disabled for ex4 planner comparison")
    _planner_mod.CuroboPlanner.__init__ = _no_curobo
    _orig_mplib_plan = _planner_mod.MplibPlanner.plan_path
    def _mplib_plan(self, now_qpos, target_pose, *a, constraint_pose=None, **k):
        return _orig_mplib_plan(self, now_qpos, target_pose, *a, **k)
    _planner_mod.MplibPlanner.plan_path = _mplib_plan
    print("REPLAY_PLANNER=mplib: cuRobo disabled", flush=True)
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
            comp = np.zeros(6)
            tcp_log = []
            joint_log = []
            last_plan = {}
            if os.environ.get("REPLAY_JOINTS"):
                _orig_lpp = env.robot.left_plan_path
                def _lpp(*a, **k):
                    r = _orig_lpp(*a, **k)
                    last_plan["left"] = (np.asarray(r["position"])[-1] if r.get("status") == "Success" else None)
                    return r
                env.robot.left_plan_path = _lpp
            prop_log = []
            if os.environ.get("REPLAY_TCP"):
                _orig_step = Scene.step
                def _logged_step(scene, *a, **k):
                    v = _orig_step(scene, *a, **k)
                    if scene is env.scene:
                        lp = env.robot.get_left_tcp_pose(); rp = env.robot.get_right_tcp_pose()
                        tcp_log.append([env.take_action_cnt] + list(lp[:3]) + list(rp[:3]))
                    return v
                Scene.step = _logged_step
            with track_execution(env, Scene) as ex:
                for t in range(1, len(prop)):
                    before = len(ex["planner_calls"])
                    target = np.asarray(prop[t], dtype=np.float64).copy()
                    if os.environ.get("REPLAY_COMPENSATE"):
                        # diagnostic only: shift position targets by the last measured target-vs-reached offset
                        target[0:3] += comp[0:3]; target[10:13] += comp[3:6]
                    env.take_action(conv.eef20d_to_ee16d(target), action_type="ee")
                    if os.environ.get("REPLAY_COMPENSATE"):
                        now = np.asarray(interface._extract_eef_proprio(env.get_obs()), dtype=np.float64)
                        comp = comp + np.concatenate([prop[t][0:3] - now[0:3], prop[t][10:13] - now[10:13]])
                    if os.environ.get("REPLAY_FRAMES"):
                        from PIL import Image
                        fd = out / "frames" / st["id"]; fd.mkdir(parents=True, exist_ok=True)
                        o = env.get_obs()["observation"]
                        Image.fromarray(np.concatenate([o["head_camera"]["rgb"], o["left_camera"]["rgb"], o["right_camera"]["rgb"]], axis=1)).save(fd / f"{t:04d}.jpg", quality=85)
                    calls = ex["planner_calls"][before:]
                    if os.environ.get("REPLAY_JOINTS"):
                        jl = np.asarray(env.robot.get_left_arm_jointState(), dtype=np.float64)
                        plan_end = last_plan.get("left")
                        joint_log.append(dict(step=t, actual=jl.tolist(), planned_end=(plan_end.tolist() if plan_end is not None else None)))
                    if os.environ.get("REPLAY_PROPRIO"):
                        prop_log.append(np.asarray(interface._extract_eef_proprio(env.get_obs()), dtype=np.float64))
                    planner.append({c["side"]: c.get("status", "EXC") for c in calls})
                    if env.eval_success:
                        break
            if joint_log:
                (out / f"{st['id']}-joints.json").write_text(json.dumps(joint_log))
            if prop_log:
                np.save(out / f"{st['id']}-proprio.npy", np.asarray(prop_log))
            if tcp_log:
                np.save(out / f"{st['id']}-tcp.npy", np.asarray(tcp_log, dtype=np.float64))
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
