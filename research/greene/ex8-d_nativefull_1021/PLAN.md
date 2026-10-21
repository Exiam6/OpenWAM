# Native full-suite reproduction after audited readiness

Question: does the released OpenWAM-Alpha-Sim-RoboTwin-Full checkpoint reproduce
93.74% clean, 93.46% randomized and 93.60% mean on the complete official endpoint?
This is evaluation reproduction, not retraining or a representation ablation.

Baseline: official upstream 782e3133f674449e2986754cfc5f7c4e204835a2 and RoboTwin
0aeea2d669c0f8516f4d5785f0aa33ba812c14b4, released checkpoint SHA256
 d07ff6f8cfb627ebf47313e65af38773fabcb3861d52fdc7c3bc5e7bee33f6fe.
Actual trusted readiness: five complete original ex8-a groups plus the complete
ex8-b recovery group, independently audited in ex8-c audit-20261021T1745Z.
30 episodes, 27 successes, 7,670 finite 20D state/action records and all videos
verified. Original failure and its partial two duplicate episodes stay retained.
Readiness rates are not official-score reproduction; exclude every pilot episode
from the new full-suite score.

Hypothesis/prediction: official native settings recover the published range.
Competing explanations for a gap: unverified numerical stack or orchestration
errors, environment differences, or released-checkpoint behavior. Full coverage,
independent native log/result/video checks and explicit gaps distinguish these
from selective reporting. This does not test RAE/DINO superiority.

Intervention relative to the frozen original supervisor: honor only the exact
expert-candidate place_object_basket rejection caught by native RoboTwin. All
other exceptions remain fatal. Official code and scientific settings are frozen.
Held fixed: all 50 official tasks, 100 accepted episodes per task per clean and
randomized mode, seed 0 native candidate sequence, unseen instructions, native
step limits, 3 input cameras, 20D state/action, unchanged checkpoint and stats,
10 synchronous denoise steps, compile on, DiT cache 0.99/max_skips3 on, full native
action horizon and cuRobo with fallback disabled. Passive debug PNG recording
returns to the official default false; keep native per-episode head videos.
Disclose benchmark torch2.7.0+cu128 vs verified2.4.1 on Blackwell. It is a software
compatibility deviation, not an exact software replication.

Metric: every task's success rate, clean/randomized equal-task means and their
mean, requiring all 100 task/mode rows =10,000 scored episodes. Missing/resource
interrupted episodes are incomplete coverage, not policy failures. Report exact
paper gaps, paired task-bootstrap intervals and exact numerical equality
separately from the original local <=2pp practical margin. That margin was chosen
locally, is not author-provided, and does not establish statistical equivalence.

Cost amendment reported in chat before registration: the initial agent-written
200 GPUh planning ceiling was not a user-imposed hard compute or spending cap.
Actual readiness projects 252.68 GPUh, or 267.68 including repeated startup and
shutdown guards. Existing user instructions authorize completing reproduction
and autonomous project progress under the idle-GPU policy. This separate phase
reserves at most 280 GPU hours (100 explicit allocations totaling279.981111h),
with seven calendar days from registration. Preserve all older phase caps and
history; this is an agent planning amendment, not a new user quote or approval.
Source: selfcheck planning-policy.json and the original reproduction request.
94 rows remain unmeasured; throughput is an estimate, not a completion guarantee.
No cloud purchase, upload or sharing authorization is added.

L0: same source and checkpoint provenance, reviewed 21 regression checks,
complete-task manifest, finite inputs, hashes, deadlines and bounded allocations.
L1-L3: audited 30-episode pilot with successful native actions, video decodes and
sampled visual review. No external whole-robot view is available. L4 first submit
only adjust_bottle clean100. Observe native server readiness, actual valid scored
rollout(s), intact native full protocol and closed full-decodable episode video.
Release the remaining99 rows only after a success among the first3 fixed episodes
with no technical classifier error. First3 all-fail or any technical failure
holds scaling for diagnosis; do not select new scenes or hide those outcomes.
This initial technical gate does not replace the required complete first100 row.

Use the existing dispatcher exclusively. gpu01 indices1/2 have any-hour priority;
all other cards only America/Chicago01:00-09:30, all require observed idle30min.
Never preempt another user's process. One bounded attempt per task/mode, no blind
retry, no resetting claims, no mid-run source or configuration changes.
Stop the affected attempt on source/hash drift, pause, fallback, unknown native
error, invalid counts/video, allocation/deadline cap or disk reserve. Diagnose
and separately register any recovery. Existing valid independent attempts retain
their own bounds. Outputs: immutable protocol/source/environment provenance,
actual commands, all outcomes/logs/videos and independently verified full result.
No representation claim until reproduction is resolved.
