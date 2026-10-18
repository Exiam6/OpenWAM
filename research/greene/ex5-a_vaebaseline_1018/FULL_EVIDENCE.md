# Matched VAE full evidence preparation

CPU postprocessing only, separate from the frozen corrected-protocol-v1 and
corrected-engine-v1 runtime. The training,
cache, evaluator, endpoints and admission registration remain unchanged. No VAE
GPU numerical result is inferred by this postprocessing. Freeze these sources separately
before starting the queue, after the first actual VAE pilot task audit exists.

There are360video pairs:180Raw/VAE primary and180Compact/VAE secondary. The
baseline is on top and VAE below. Numerical effects are baseline minus VAE;
left-only success favors the baseline. All scenes, seeds and failures stay.
The browser labels one-sided successes by model name (Raw, Compact or VAE),
because the paired camera views are stacked top/bottom. Internal outcome values
and numerical signs remain unchanged; the shared filter calls the top model
Baseline.
The45primary pilot videos are reused only after their complete frozen pilot media
manifests exist and all result/video/preview hashes match. Other pairs encode the
three policy cameras for both sides, aligned by action index with final-frame
hold. There is no external whole-robot view and no inferred exhaustive review.

The CPU queue uses at most two encoder children, two ffmpeg threads each,180s
per child, and the original17:00UTC deadline. It has no CUDA imports, no model
forwards, no GPU allocation and no automatic retries. Incomplete media or failed
processes are retained. Current numerical GPU code is never edited by this work.

The final report requires the complete fixed180VAE evaluations, the complete
corrected baseline, all nine training/evaluation parent claims, complete preflight,
all360hash-verified decoded videos, and18actual first-scene contact-sheet reviews
(two comparisons x three seeds x three tasks, six sampled indices per sheet).
Nine primary reviews may be inherited from the actual pilot reviews only after
matching the same preview hash and sampled indices. Decode is not visual review.

The archive includes the small evidence files behind every release: cache and
smoke results, tiny-fit metrics, deployment action traces, per-seed evaluation
releases and budgets, pilot media and visual reviews, parent results, reference
actions and independent episode audits. A gate-evidence index maps original
paths to archive paths and SHA256 hashes. Large model/cache binaries remain
external artifacts identified by the registered/result hashes.

The final archive also retains the registered CPU pilot queue, immutable plan and
source hashes, test evidence, launch record, completed per-seed parent results and
logs. Each queue result must match the actual 15-episode pilot and its successful
parent claim. The independent final-training audit and its source/observer-only
correction history are included, with final model identity and training-file
hashes rechecked against the actual outputs. The completed CPU media process and
its original source are retained. These additions change archive provenance only;
the frozen training, evaluation, video generation and statistical endpoints stay
unchanged.

The archive also retains the original seed42/43 remaining-endpoint budget
refusals, their measured cost observations, and the immutable pending amendment
proposal with its referenced evidence hashes. These records remain refusals and
a proposal: collecting them grants no launch authority and changes no cap.
The complete report still requires the original releases and per-seed budget
checks. Any later authorized amendment needs a separately verified implementation
and explicit report provenance; this archive-only addition does not supply it.

The full-readout primary is Raw minus VAE; Compact is a secondary dimensionality
control, never a replacement primary. Use the unchanged10000crossed bootstrap,
RNG921018, and registered5pp/two-positive-seeds/positive-lower-bound criterion.
Report measured per-stage occupation, data/training/model differences and the
known-development-cohort limitation. Equal data/steps do not mean equal FLOPs.
The larger official policy is a functional control, not part of this comparison.
