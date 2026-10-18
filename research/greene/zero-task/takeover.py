"""ex4b takeover: replay the expert (with offset compensation) to the pre-grasp state, then let the policy act.

During the replay phase the policy client is stepped on every observation exactly as in a rollout (its outputs are ignored),
so its history matches a real episode. Usage (benchmark-env python):
  takeover.py <port> <output_dir> <lead_steps> <demo_dir> [<demo_dir> ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

STUDY = Path(os.environ.get("REPLAY_HELPERS", Path(__file__).resolve().parent))
sys.path.insert(0, str(STUDY))
from migration_render import configure_robot_paths, pin_renderer  # noqa: E402

pin_renderer()
sys.path.insert(0, os.environ.get("ROBOTWIN_BENCH_DIR", "/home/zifanz4/openwam-runtime/assets-source/temporal-20260926/OpenWAM/benchmarks/robotwin"))
from eval_policy_wrapper import bootstrap_robotwin_module  # noqa: E402

mod = bootstrap_robotwin_module()
configure_robot_paths()
import openwam2robotwin_interface as interface  # noqa: E402
import yaml  # noqa: E402

conv = interface.action_conversion
INSTR = {"handover_block": "Handover the block.", "place_object_basket": "Place the object into the basket.", "adjust_bottle": "Adjust the bottle."}


def main():
    port, out, lead = int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    args = yaml.safe_load(Path("task_config/demo_clean.yml").read_text())
    emb = yaml.safe_load(Path("task_config/_embodiment_config.yml").read_text())["aloha-agilex"]["file_path"]
    ec = mod.get_embodiment_config(emb)
    camera = mod.get_camera_config(args["camera"]["head_camera_type"])
    args.update(task_config="demo_clean", embodiment_name="aloha-agilex", left_robot_file=emb, right_robot_file=emb,
                left_embodiment_config=ec, right_embodiment_config=ec, dual_arm_embodied=True, head_camera_h=camera["h"],
                head_camera_w=camera["w"], need_plan=True, save_data=False, render_freq=0, eval_mode=True, is_test=True,
                eval_video_save_dir=None)
    interface._STEP_LIM_OVERRIDES = {}
    model = interface.ModelClient(host="127.0.0.1", port=port, send_state=True, state_dim=20, request_timeout=120, action_type="ee")
    cur = {"step": 0, "seed": 0, "action": None}

    def send(payload):
        payload = dict(payload)
        payload.update(_study_step=cur["step"], _study_action_seed=cur["seed"])
        r = model._client.predict_once(payload)
        cur["action"] = np.asarray(r["action"])
        return r

    model._client.predict = send
    rows = []
    for d in sys.argv[4:]:
        d = Path(d)
        st = json.loads((d / "status.json").read_text())
        prop = np.load(d / "full-trajectory.npz")["proprio"]
        task, seed = st["task"], int(st["seed"])
        close = int(np.argmax(prop[:, 9] < 0.5))
        handoff = max(1, close - lead)
        args["task_name"] = task
        args["save_path"] = str(out / f"{task}-{seed}")
        env = mod.class_decorator(task)
        row = dict(id=st["id"], task=task, seed=seed, expert_first_close=close, handoff_step=handoff)
        t0 = time.monotonic()
        try:
            env.setup_demo(now_ep_num=0, seed=seed, **args)
            env.set_instruction(instruction=INSTR[task])
            interface.reset_model(model)
            cur["seed"] = int(np.random.SeedSequence([seed, 42, 811]).generate_state(1, dtype=np.uint32)[0])
            comp = np.zeros(6)
            native = type(env).take_action

            def compensated(action, action_type="qpos"):
                a = np.asarray(action, dtype=np.float64).copy()
                cmd = a.copy()
                a[0:3] += comp[0:3]
                a[8:11] += comp[3:6]
                native(env, a, action_type=action_type)
                reached = np.asarray(interface._extract_eef_proprio(env.get_obs()), dtype=np.float64)
                comp[0:3] += cmd[0:3] - reached[0:3]
                comp[3:6] += cmd[8:11] - reached[10:13]

            env.take_action = compensated
            lg, rg = [], []
            # phase 1: expert to pre-grasp, policy observes (outputs ignored)
            for t in range(1, handoff + 1):
                cur["step"] = int(env.take_action_cnt)
                obs = env.get_obs()
                o = obs["observation"]
                example = {"cams": {"head": o["head_camera"]["rgb"], "left": o["left_camera"]["rgb"], "right": o["right_camera"]["rgb"]},
                           "lang": INSTR[task], "state": interface._extract_eef_proprio(obs)}
                model.step(example)
                env.take_action(conv.eef20d_to_ee16d(np.asarray(prop[t], dtype=np.float64)), action_type="ee")
            # phase 2: policy acts
            while env.take_action_cnt < env.step_lim and not env.eval_success:
                cur["step"] = int(env.take_action_cnt)
                interface.eval(env, model, env.get_obs())
                a = np.asarray(cur["action"])
                lg.append(float(a[9])); rg.append(float(a[19]))
            lg, rg = np.array(lg), np.array(rg)
            row.update(success=bool(env.eval_success), steps=int(env.take_action_cnt),
                       policy_left_close_after=(int(np.argmax(lg < 0.5)) if (lg < 0.5).any() else None),
                       policy_right_close_after=(int(np.argmax(rg < 0.5)) if (rg < 0.5).any() else None),
                       comp_final_mm=[round(1000 * x, 2) for x in comp])
        except Exception as exc:
            row.update(error=repr(exc))
        finally:
            env.close_env(clear_cache=True)
        row["seconds"] = round(time.monotonic() - t0, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    model._client.close()
    (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
