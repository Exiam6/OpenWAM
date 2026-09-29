# StageC: original-policy trajectory repeatability, 2026-09-21

Frozen before any new closed-loop outcomes. This stage diagnoses execution
repeatability; reused scenes and two repeats do not estimate general success rates.

## Fixed design and budget
12 rollouts total: rounds0/1, each clean then noise sigma=.20, each with the same
ordered scenes400000,400001,400002 from repeatability-20260921/scenes.json. Preserve
all original instruction strings and ten-way NumPy draw consumption. No expert
refiltering. Start a fresh original-policy server for each round; reset via the
native client at every episode boundary. Same frozen model, BF16, seed42 for each
generation,10 denoising steps, compileoff, original cache settings, full32-action
execution horizon,400-action episode limit. No representation intervention.

At most one L40S on cm001 GPU2 and the existing gpu-cm001-2.lock. Three strict idle
samples before each round: memory<100MiB/utilization0/ECC0/no compute jobs. A busy
GPU defers the stage without disrupting others. Original90-minute wall allocation
started at repeatability-20260921/launch.json, shared with stagesA/B. Compute and
obey only remaining time; no automatic extension. Require4GiB free disk for logs.

## Passive tracing
Use the unchanged norm_client/robustness_client/native RoboTwin control loop.
Wrap observation transformation, transport predict and environment take_action
without changing their arguments, returned objects, action types, RNG states or
model settings. Record per step:
- raw three-camera hashes, EEF20D and joint vector; altered policy camera hashes;
- canonical native JSON request hash and full request in a local gzip file;
- server20D action, server counter,16D target delivered to environment.take_action;
- post-call action count and success flag. These16D targets are not measured
  physical joint trajectories. Keep native full videos for all12 episodes.
Assert initial head/state/prompt matches the frozen manifest, and each initial
complete payload matches the six already verified saved requests. Do not require
action equality to historical data as an admission gate. Require exactly one
request and one environment action per callback; server_step must equal step+1
(no unnoticed retry/buffer advancement). Retain incomplete attempts on failure.

## CPU checks before launch
Exercise the actual native interface.eval with a fake transport/environment:
tracing vs untraced must pass identical action values/type and preserve object
identity at each wrapper boundary; RNG state must match. Exercise the actual
upstream replay loop for3 scenes and ten-choice RNG consumption, including400-step
limit delegation, and verify gzip request/hash roundtrip. Tests must exit0 freshly;
stale result files are insufficient. Freeze source hashes before GPU use.

## Prespecified analysis
For each of6 condition/scene pairs compare round1 with round0 over their shared
prefix, reporting independently the first zero-based step with different:
raw camera bytes, EEF state bytes, joint vector bytes, complete policy request,
server action20D, and environment target16D. Record camera-specific differences.
Also compare episode lengths/outcomes, full-horizon equality, numeric max absolute
EEF/action differences, and first-action equality to the saved stageB identity
request result. Do not silently discard a pair with different outcomes or lengths.

Inspect differences chronologically. A raw-image or state divergence before the
first different action supports an execution/observation origin for that pair,
but does not locate the physics/rendering/IK cause. A different action on the same
request requires checking buffer position and full generation context; this alone
is not proof of model nondeterminism. Preserve every comparison, no selected seeds.
Publish all12 videos, complete lightweight traces and the first-divergence request
examples. Full per-step PNG payloads stay local to bound public report size.
No new training, model selection, adapter fitting, PR submission or real hardware.
