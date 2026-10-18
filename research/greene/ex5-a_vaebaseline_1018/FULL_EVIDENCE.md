# Matched VAE full evidence preparation

CPU postprocessing only, separate from frozen GPU runtime de4cf44. The training,
cache, evaluator, endpoints and admission registration remain unchanged. No VAE
GPU numerical result exists yet. Freeze these postprocessing sources separately
before starting the queue, after the first actual VAE pilot task audit exists.

There are360video pairs:180Raw/VAE primary and180Compact/VAE secondary. The
baseline is on top and VAE below. Numerical effects are baseline minus VAE;
left-only success favors the baseline. All scenes, seeds and failures stay.
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

The full-readout primary is Raw minus VAE; Compact is a secondary dimensionality
control, never a replacement primary. Use the unchanged10000crossed bootstrap,
RNG921018, and registered5pp/two-positive-seeds/positive-lower-bound criterion.
Report measured per-stage occupation, data/training/model differences and the
known-development-cohort limitation. Equal data/steps do not mean equal FLOPs.
The larger official policy is a functional control, not part of this comparison.
