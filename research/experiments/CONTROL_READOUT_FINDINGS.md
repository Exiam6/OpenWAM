# Restoration/control-readout diagnostic — completed 2026-09-22

Frozen source6687e266a985125b4d7e2ecf8f36222c67e60933. All10 prespecified
conditions retained, exit0. One bounded CPU run,2threads,CUDA invisible; fit two
ridge readouts on420 unique clean observations from35 episodes. Alpha selected
only by five training-episode folds: visual+proprio1, proprio-only0.01.

On60 cached validation observations from5 previously used episodes, the fixed
seed42 adapter reduces full latent restoration MSE by34.72%. Its noise-condition
translation normalized readout MSE falls11.84%, but gripper rises5.82%; overall
8-coordinate mean falls11.04%. On clean input, total error increases2.64%, with
translation+2.64% and gripper+2.51%. The original seed43/44 sensitivity checks show
the same directions: noisy translation-12.09%/-12.99%, gripper+10.45%/+6.95%;
clean total+3.14%/+1.56%. These are correlated development comparisons, not new
independent confirmation or three new tasks. No alternative seed selected.

The primary noisy-gripper paired episode-bootstrap interval includes zero.
All per-episode values and intervals retained; bootstrap resamples5episodes,
not60windows. Do not claim a proven gripper degradation or broad significance.

Proprio-only gripper normalizedMSE0.03729 is lower than visual+proprio clean0.07744.
Thus the group tradeoff may reflect the fixed linear readout's regularization,
capacity or nuisance sensitivity. It cannot isolate irreversible information loss
in S-VAE. Clean visual+proprio translation0.46757 versus proprio-only0.61977 and
mismatched visual0.58381 supports task-useful visual readout in this setup only.
These values are achieved-state forecasts, not commands, contact or policy success.

No further adapter loss sweep follows from these results. The predeclared
consistent translation-and-gripper benefit condition was not met. Keep the
published inference defaults. Contribution: document the reconstruction/control
metric gap, report groups and proprio baselines, and make reproduction scripts
available independently of the237-line reconstruction evaluator.

Software gate also closed: normal documented S-VAE CLI succeeded on an existing
trained checkpoint and3 fixed feature clips packaged in native shard format,
CPU batches2and1. Metrics match within1e-6, checkpoint normalization preserved,
input hashes unchanged. Existing policy-env used unchanged:torch2.7.0+cu126,
transformers5.17.0,diffusers0.40.0. Full upstream suite not run. Fixture is a
software check, not a new held-out scientific estimate.

Cache alignment:480clean/900noisy mappings and rawframe+4 labels checked;35train
and5val episodes disjoint, no demonstration test episodes opened. Original
extraction lacked per-image hashes, so retrospective feature byteprovenance is
not proven. Current encoded-image/label hashes and source checks retained.

Audit independently recomputed all10 prediction groups and paired bootstrap
intervals from saved predictions, verified train-only target scaling,11 frozen
inputs/sources unchanged, and selection timestamp before scoring. No refit,
new GPU job,80-rollout rerun or stopped-context retry occurred.

Next contribution step: package this development counterexample with the evaluator
review, including the successful normal CLI gate and readout limitations. A
scientific claim of improvement still needs a separately frozen fresh-episode
study and closed-loop validation; current results do not justify automatic
additional fitting on the same development episodes.
