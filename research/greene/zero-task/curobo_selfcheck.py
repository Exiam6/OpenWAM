"""ex4: cuRobo self-consistency on this GPU (no simulator).

Builds the same MotionGen as RoboTwin's CuroboPlanner (aloha-agilex left arm config, table world),
samples joint configurations, takes cuRobo's own FK pose as goal, plans from a perturbed start, and
measures cuRobo's own FK error of the plan end and the reported position/rotation error.
"""
import json
import sys

import torch
sys.path.insert(0, "/home/zifanz4/OpenWAM/benchmarks/robotwin")
from eval_policy_wrapper import _patch_warp_torch_namespace
_patch_warp_torch_namespace()
from curobo.geom.types import WorldConfig  # noqa: F401
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

yml = "/home/zifanz4/openwam-runtime/migration-20260927/robot-configs/curobo_left.yml"
world = {"cuboid": {"table": {"dims": [0.7, 2, 0.04], "pose": [0.0, 0.0, -0.05, 1, 0, 0, 0.0]}}}
cfg = MotionGenConfig.load_from_robot_config(yml, world, interpolation_dt=1 / 250, num_trajopt_seeds=1)
mg = MotionGen(cfg)
mg.warmup()
torch.manual_seed(0)
dev = mg.tensor_args.device
names = mg.kinematics.joint_names
q_home = torch.zeros(len(names), device=dev)
rows = []
J = json.load(open("/home/zifanz4/openwam-runtime/zero-task-ex4/replay-joints/handover_block-seed4101100-joints.json"))
print("joint names", names, flush=True)
n = len(names)
for k in range(1, len(J), max(1, len(J) // (int(sys.argv[1]) if len(sys.argv) > 1 else 20))):
    q_goal = torch.tensor(J[k]["actual"][:n], device=dev, dtype=torch.float32)
    q_start = torch.tensor(J[k - 1]["actual"][:n], device=dev, dtype=torch.float32)
    ks = mg.compute_kinematics(JointState.from_position(q_goal.view(1, -1), joint_names=names))
    goal = Pose(ks.ee_pos_seq.view(1, 3), ks.ee_quat_seq.view(1, 4))
    res = mg.plan_single(JointState.from_position(q_start.view(1, -1), joint_names=names), goal, MotionGenPlanConfig(max_attempts=10))
    row = dict(step=J[k]["step"], success=bool(res.success.item()), status=str(res.status))
    if row["success"]:
        end = res.interpolated_plan.position[-1].view(1, -1)
        ke = mg.compute_kinematics(JointState.from_position(end, joint_names=names))
        row["fk_pos_err_mm"] = float(1000 * torch.linalg.norm(ke.ee_pos_seq.view(3) - goal.position.view(3)))
        row["joint_err_max"] = float((end.view(-1) - q_goal).abs().max())
    rows.append(row)
    print(json.dumps(row), flush=True)
ok = [r for r in rows if r["success"]]
summary = dict(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)), torch=torch.__version__,
               plans=len(rows), successes=len(ok),
               fk_pos_err_mm_max=max((r["fk_pos_err_mm"] for r in ok), default=None),
               fk_pos_err_mm_median=sorted(r["fk_pos_err_mm"] for r in ok)[len(ok) // 2] if ok else None)
print("SUMMARY", json.dumps(summary), flush=True)
