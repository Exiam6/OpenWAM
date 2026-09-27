# End-to-end representation validation

New user authorization: 完整验证是否新表征能够怎么贡献到原来的openwam 把实验跑全 改训练就训练 完整验证.
This is a new study, not an extension or resumption of any expired cohort.

## Questions and controlled comparisons
1. Does DINO-PCA48 preserve native OpenWAM control quality compared with DINO-SVAE48, and improve robustness over Wan48?
2. Do benefits survive three paired policy-training seeds, fresh simulator scenes, sensor noise and camera pose changes?
3. Are translation, rotation and gripper/contact-stage errors consistent with any success-rate difference?

The practical primary comparator is OpenWAM's native DINO + S-VAE path, not only its Wan VAE alternative. DINO-PCA is a parameter-free bottleneck with training-only fitted statistics; it has no pixel decoder and is not a complete RAE. Do not label an offline probe improvement as policy improvement.

## Stages and decision rules
A. Independently audit the already completed 60-scene offline confirmation. Done: 60 raw scenes, 135 prediction conditions and 99 paired intervals checked without refits/inference; all numerical gates passed.
B. Add an optional DINO-PCA encoder in an isolated source copy. Verify projection arithmetic, frozen-stat precision, temporal causality, checkpoint reload and native deployment parity. Native S-VAE and Wan implementations stay intact.
C. Profile matched reduced-size native OpenWAM training using training data only. Use native dual-system joint self-attention, causal first frame, eef20, same action normalization and same 105 train episodes. Three routes: Wan48, DINO-SVAE48, DINO-PCA48. Same 48 latent channels and spatial/temporal geometry. Copy identical common trainable initialization across routes for each seed (42,43,44); frozen encoders can otherwise consume different RNG counts.
D. Before collecting new policy outcomes, freeze exact optimizer updates, batches, schedule, noise recipe, scene seeds, episode count, rollout horizon, inference steps, metrics and all stopping criteria. Throughput may set a feasible shared budget; held-out success may not select a winning route or checkpoint. Final fixed-step checkpoints are the primary comparison.
E. Run all three routes and seeds. Report every run, task, condition and failure, including no benefit. Fresh closed-loop scenes must not overlap old 80 rollouts, exposed layer cohorts, expired 56 scenes or audited independent 60. Pair scene/action-noise seeds across methods. Confidence intervals must respect scene and training-seed clustering. Record grasp/contact diagnostics as measured proxies, not force sensing unless simulator contact data actually exists.
F. If the controlled experiment supports a useful change, prospectively fix matched full-size adaptation to test scale transfer. A reduced-size result is not a reproduced 5B improvement. If all methods fail to learn, report that the test cannot rank representations and diagnose training before claiming any result.

## Resources and bounds
At most 8 concurrently owned GPUs, initially plan within 576 GPU-hours and a 7-day review horizon. Every launch needs an idle GPU/UUID/process/ECC check and a per-GPU advisory lock; use only bounded jobs and recheck pause markers. No unrelated process termination, no new agents or timers. Model/data/checkpoint outputs use /data02 (not nearly full /home); check free space before checkpoints. Small manifests and reports stay in this repository. Old deadlines/protocols/raw results remain unchanged.

## Contribution targets
An optional checkpoint-portable PCA encoder, reusable matched-representation training/evaluation configuration, component-aware robustness diagnostics, and an evidence-backed model choice. Existing negative results are included. A claim of improved full OpenWAM requires successful native training AND fresh closed-loop evidence; probe-only claims remain limited.

## Publication
Update the existing report locally and attempt the already authorized public destination when available. The last verified Sites response was project-not-found (404); no publication success is claimed.
