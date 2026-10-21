# Ten-task dataset feasibility pilot

Standing rules: /home/zifanz4/research_lab/TOP_PRIORITY/.

Question: can the same frozen simulator and three policy cameras produce valid,
timestamped expert trajectories on all ten preselected tasks?

This is a data-readiness experiment, not a representation comparison or evidence
of policy success. The task identity varies; collector, camera layout, physics,
action/state convention and seed-selection rule remain fixed. A competing
explanation for low downstream success is an invalid or incomplete training set.

The first five fixed candidates gate continuation: at least one must pass native
expert success and trajectory checks. Then collect the remaining twenty fixed
candidates. Each task has twenty training and five development candidates.
No final-test data is generated. Seeds are disjoint from previous studies.
Failed candidates remain; there is no replacement by a favorable seed.

Primary output: valid trajectories per task with timestamps, finite 20D states,
three-view images, native expert success, and retained failures. A valid trajectory
has at least 33 frames. The five development candidates never train normalizers
or models. Check physical time intervals before creating action labels.

Each task is one independent bounded job, at most 6,000 seconds; ten tasks reserve
16.67 GPU hours. Each candidate has a 180-second in-process timeout and a
220-second process timeout. Resource interruption stops only owned processes.
Deadline and exact inputs are frozen in protocol.json before dispatch.

Support: valid train/development trajectories on every task and correct image /
state / time contracts. Failure: any task has no feasible candidate or corrupt
data. Neither outcome establishes DINO superiority. Training requires a separate
registered L1/L2 model-readiness check; this pilot does not automatically launch
full-scale training. A successful collection is followed by a dataset audit,
paired feature caches, and cost/power estimates for the representation study.
