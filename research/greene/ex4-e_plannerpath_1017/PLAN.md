# ex4-e_plannerpath_1017: does configuration relocation disable ALOHA calibration?

Registered before GPU replay. Research question: establish a valid RoboTwin
execution path before interpreting RAE policy comparisons.

Baseline: 7b1a7a25 replay_expert_ee.py and existing ex4a replay-gpu01 records
(handover 0/5, adjust 2/2, place 0/1). Local migration_render.py redirects cuRobo
configs into migration-20260927/robot-configs, removing 'aloha-agilex' from the
path. RoboTwin envs/robot/planner.py branches on this substring in BOTH plan_path
and plan_batch, applying z rotations -0.02/-0.01 rad only when it is present.

H: the relocation silently skips native ALOHA calibration and causes the fixed
pose offset and expert replay failures. Alternative: numeric planner/physics
differences dominate, and restoring the branch does not restore execution.

Single intervention: identical YAML bytes reachable through a directory whose
name contains aloha-agilex, preserving the native conditional branch. Copy or
symlink configs into this experiment only; no running code, original config,
RoboTwin checkout, physics, model or environment package is modified. Do not use
ex4c accumulated offset compensation. All targets are the same expert states.

L0: execute the actual planner source AST through the goal-transform assignment
on CPU, for both arms, comparing old relocated path, alias path and native path;
check alias equals native and differs from old by the registered rotations.
Verify original and alias YAML hashes are equal. No CUDA needed for this gate.

L1: paired old-path and fixed-path replay on handover seed4101100, recording
proprio and every rendered step. Require initial proprio match within 1e-6 and
no infrastructure/planner failures. Old-path outcome must reproduce prior fail;
fixed-path success and reduced static-grasp error support proceeding to L3.
If fixed fails, preserve logs and diagnose instead of launching the full set.

L3: all nine existing dev demonstrations (adjust 4100100/101, handover
4101100..104, place 4102100/103), each under both paths; reuse paired L1 rows.
Primary native success by task; secondary end-effector tracking on stationary
target spans and planner failures. Prediction: fixed handover >=4/5 vs old <=1/5,
adjust >=1/2 fixed. Place is exploratory. Those counts support H for handover;
failure to reduce the pose error refutes this mechanism; otherwise mixed.
This cannot prove learned-policy improvement or cross-host equivalence.

Budget: <=1 GPU-hour, <=60 minutes active wall, deadline 2026-10-18 12:00 CDT.
One additional simulator on an authorized GPU1 or GPU2, only own-user occupants,
>=30 GiB free, initial utilization <=85%; four CPU threads. Physics remains CPU.
Avoid GPU3 because another user has occupied it; existing GPU4 diagnostic queue
continues. Every job has immutable code, environment, input hashes and log.
Stop on pause, time limit, mismatched initial state or infrastructure failures.
No overwrite/retry of a failed attempt; result-gated progression only.

Outputs: CPU branch audit, paired replay JSON/proprio/frames, paired video,
scoped conclusion and a proposed robust path-remapping fix if supported.
