# Measurement semantics clarification (post-launch source inspection)

2026-09-21 23:03 CDT. No frozen protocol, runtime source or experiment changed.
The saved `joint_state` field is the original observation `joint_action.vector`.
Pinned RoboTwin envs/_base_task.py:487-494 assembles it from
get_left_arm_jointState/get_right_arm_jointState. In envs/robot/robot.py:494-506,
those methods return joint.get_drive_target(), plus commanded gripper values.
They are **joint drive targets**, not measured physical joint positions. Separate
get_left_arm_real_jointState/get_right_arm_real_jointState methods exist but were
not recorded. The published report labels this field accordingly.

The EEF vector is assembled from observed endpose and commanded gripper fields;
its gripper dimensions are not measured finger opening/contact labels either.
The16D take_action input is an end-effector command, not an executed joint path.

Therefore identical joint_state hashes cannot establish identical physics state.
The first clean pair has tiny end-effector pose differences after the same first
action, while drive-target values remain equal. This narrows the diagnostic but
does not isolate physics, rendering, IK or another component. Preserve every pair.

Both inspected files match the unmodified pinned RoboTwin commit
0aeea2d669c0f8516f4d5785f0aa33ba812c14b4 byte-for-byte. The frozen protocol requested
the native joint vector; this note clarifies its interpretation after inspection.
