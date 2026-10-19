# GPU-stage cost and admission draft

This is a planning artifact, not a launch amendment. Current registry GPU budget
remains zero. The data/model CPU contracts now pass, but no encoding, gradients
or policy forward has been validated for the new arm.

Historical evidence: the six original compact/raw native seed42/43/44 runs each
completed20000 steps in about1.44–1.53 occupied GPU-hours, with peak allocated
memory about16.4GiB. Exact source values and hashes are in
`/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/historical-training-cost.json`.
The new frozen VAE has different memory/runtime costs. These historical numbers
justify measuring feasibility, not assuming the VAE will meet the same timing.

Candidate hard budget: at most20.5 occupied GPU-hours, separate from G/H:
2hours for VAE cache/encoding validation,0.5hours for tiny gradient/overfit and
serialization checks,3hours for each of three fixed20000-step runs, and3hours
for each of three60-scene native evaluation runs. Loading, smoke, analysis that
occupies a GPU and cleanup must count inside each cap. No cap extension or
retry is implied. The full staged gate criteria and implementation must be
committed before replacing the current zero-GPU registration.

Prefer truly idle extra GPUs during the already authorized01:00–10:00 CDT window
(06:00–15:00 UTC on October18). Do not preempt another user or count a zero-util
GPU with resident foreign processes as idle. Keep current G/H work within its
own registered limits. No same-GPU sharing with G/H without a separate measured
capacity/throughput decision. The existing noon deadline remains17:00 UTC.

Before admission require the complete G pilots and completed H paired control
with positive corrected handover, their audits and actual video review. Before
full training require strict VAE first-frame causal/encoding/cache-deployment
agreement on preregistered training windows, finite gradients and native action
loss/forward consistency, a tiny fitting gate on training inputs only, and a
measured complete-stage projection. Do not substitute flow loss for closed-loop
success, or use development success to select a checkpoint.

Worst-case serial cache+tiny then parallel three-seed training then parallel
three-seed evaluation needs8.5hours plus a15-minute per-seed video-review allowance if three idle GPUs are available throughout.
The09-hour extra-GPU window could fit with a small reserve, but availability is
not guaranteed. Use actual remaining time and measured stage costs before each
admission; if the complete registered endpoint cannot fit, preserve preparation
instead of silently reducing seeds, steps or scenes. Do not promise an overnight
result before real cost and resource gates have passed.

Evaluation implementation preparation is detailed in EVALUATION_PREPARATION.md: each seed uses15pilot plus45remaining evaluations under its unchanged3-hour cumulative cap; allthree seeds and180evaluations remain required. Phase progress is independent per seed to avoid idle barriers.
