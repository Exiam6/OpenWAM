# Matched VAE staged GPU admission

This amendment supersedes the zero-GPU restriction of the static preparation
stage only after the committed source is frozen and gpu-registration.json is
created by register_gpu.py. It does not claim that any VAE GPU stage has passed.
The scientific question, prediction, metric and full endpoint remain PLAN.md.
The previously requested ex5 Wan-VAE control now has the required evidence that
the corrected environment can carry a handover and placement signal.

## Evidence and controls

The complete corrected-factorial pilots passed their registered audit/cost gate
(ex4-g/L3-release.json). The official functional control is complete and supports
calibration transfer (ex4-h/final-bundle.json, its hashed complete video bundle,
success-summary.json and actual pilot visual review). Its larger policy is not
included in the matched comparison. Ex4-f supplies the paired Raw pilot videos;
the complete independently audited ex4-g Raw and compact endpoints supply the
final comparison. No final VAE comparison is released until that baseline and
all three VAE seeds are complete.

The restored input audit proves all 6424 window labels, train/development split,
RGB indices and three minibatch schedules. CPU model construction matches every
same-shaped trainable tensor to its original per-seed initialization and copies
only frozen VAE/text components from pretrained assets. CPU tests do not prove
GPU encoding, gradients, rollout correctness or throughput.

## Fixed scope, caps and allocation

Raw DINO768 -> pretrained native Wan VAE48 is the only scientific intervention.
Train the same 12-layer,1024-wide video trunk and512-wide action model for20000
steps on each of seeds42/43/44; batch3 with accumulation2, native shifts5,
unchanged optimizer, labels, normalization and minibatches. There is no import of
trained official policy weights, checkpoint selection or tiny-checkpoint reuse.

Total occupied GPU cap is20.5hours, separate from all ex4 budgets:

| Stage | Hard occupied cap | Required next-stage evidence |
| --- | --- | --- |
| Cache and encoding | 7200s | All6424 windows exact; three training-window causal/native encoding checks; sampled cost fits |
| Smoke, deployment, tiny fit combined | 1800s | Three-step finite nonzero gradients; strict reload; native/cache/action consistency;512-step training-only fit and runtime projection |
| Fixed full training | 10800s per seed, three seeds | Same fixed20000-step endpoint, complete checkpoint and hash |
| Native evaluation | 10800s per seed, three seeds | First15pilot pairs then45remaining; audit, actual video review and measured cost gate |

The pilot part is additionally capped at3600s within each evaluation cap.
Parent process lifetimes charge imports, loading, checks, smoke and cleanup.
No budget extension or automatic retry. Scientific failures are retained as valid
outcomes; infrastructure failures preserve partial artifacts and stop that attempt.
A failed preflight stops all dependent work. A failed full seed is retained; other
already running seeds finish within their own caps, but no complete comparison is
claimed. Budget or allocation refusal does not silently reduce steps or scenes.

Primary scheduling uses truly idle extra GPUs5/6/7 on gpu-01 during the existing
01:00–10:00CDT window on2026-10-18 (06:00–15:00UTC). GPU0 may be used only if
truly idle; it currently has foreign work and also has a conditional G dispatcher.
GPUs1–4 remain G's current resources; this amendment does not preempt their work.
No same-GPU sharing. A GPU is eligible only with no resident compute process,
zero utilization and under256MiB used on three polls and a final check. Three
eligible extra GPUs are required at initial admission. Future availability is
not guaranteed and is rechecked before each phase.

Worst-case serial cache+tiny then parallel training and evaluation requires8.5h,
plus15min for per-seed video review and cleanup reserves. The initial parent
requires this full envelope to fit the remaining allocation; after tiny fitting,
its slowest of eight tail intervals projects full training with900s overhead,
which must fit10740s. After each pilot, full native action caps for the remaining45
scenes must fit the remaining per-seed budget using the EVALUATION_PREPARATION.md
cost rule. The global deadline remains2026-10-18T17:00:00Z. Extra-GPU work stops
before15:00UTC even if that global deadline is later.

## Frozen evidence, launch and reporting

register_gpu.py verifies the inherited1184-file engine/benchmark/scene inventory,
new evaluation templates and robot configs, common training helper, all new source,
model templates, recovered trajectories, old cache, per-seed initialization and
frozen model assets. Source is committed and the runtime code directory is created
once, without overwrite. Parent admission rechecks assets; children recheck their
registered code and their generated checkpoint/cache evidence. No running code
is edited. All pause paths remain honored.

The v7 compute monitor includes VAE cache windows, training steps, actual parent
and active-child PIDs, per-scene evaluations and GPU ownership. It preserves the
existing half-hour cadence. It observes work; it does not bypass any stage gate.
The initial CPU command is code/run_preflight.py --gpu5 at or after06:00UTC.
Passing it does not launch full training. release_training.py recomputes actual
criteria; run_batch.py then starts all three fixed seeds on separate idle GPUs.
Each completed seed independently advances to pilot, CPU paired videos, actual
visual review, release_evaluation.py and remaining45 evaluations.

readout.py checks result/trace hashes and paired state/noise/task conditions.
Its primary is Raw minus VAE; compact minus VAE is secondary descriptive only.
The fixed10000 crossed seed/within-task-scene bootstrap uses RNG921018. All540
matched rows across the three arms must be present before formal comparisons.
All result reports include paired videos, per-seed values, compute cost and the
limitations in PLAN.md; no unseen-benchmark or broad population claim follows.
