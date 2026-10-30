# RAE visual representation study

**Execution update, 2026-10-30:** experiments are paused by user instruction.
The [new execution plan](EXECUTION_PLAN_20261030.md) supersedes the operational
sequence and expired deadlines. This file preserves the original scientific
registration and methodological context; registered endpoints are not deleted.

Source: OpenWAM `782e3133f674449e2986754cfc5f7c4e204835a2`.
Worktree: `research/rae-visual-20261023`. The released RoboTwin-Full checkpoint
and its 10,000-episode native evaluation remain the reproduction baseline.

## Question and competing explanations

Does replacing the reconstruction-trained Wan video VAE latent with a frozen,
pretrained visual representation improve closed-loop RoboTwin success when the
training data, model trunk, action head, optimizer, budget, and evaluation are
matched? Apparent gains could instead come from a wider DiT input/output head,
different latent scale or noise schedule, changed token count, extra decoder
training, or different demonstrations. Record and control each difference.

## Proposed method

1. Start with the public `facebook/dinov2-base` snapshot at
   `f9e44c814b77203eaa57a6bdbbd535f21ede1415` (model SHA256
   `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`).
   The DINOv3 candidate requires manual access approval. Freeze DINOv2,
   preserve the three RoboTwin camera order and frame timing, pad only the
   patch border, and resample its 14-pixel grid to the released model's
   16-pixel latent grid. Record that spatial resampling as a method choice.
2. Fit per-channel latent normalization on the **training split only**. The
   7,985-frame stride-16 pass yielded statistics at
   `/root/zifan/openwam-runs/rae_feature_stats_stride16.pt`, SHA256
   `c582c0c8dca54f0944d4d5a99dd1fc34a25948bd61313ba17ba61fd92d54b838`.
   Raw channel means span -5.407 to 4.874 and standard deviations span
   0.122 to 6.419. The calibrated DINOv2 path stores the mean/std as frozen
   checkpoint buffers; the earlier per-token normalization stays a separate
   ablation. Do not change normalization in a resumed run.
3. Train a pixel decoder from frozen DINO patch features on training videos to
   make a representation autoencoder. Validate reconstruction on held-out
   demonstration videos. The decoder must not consume actions or test frames.
4. Adapt the OpenWAM DiT input/output projections to the representation's
   channel and patch shape, keeping the common trunk and action head as equal
   as possible. Initialize both matched VAE and RAE runs from the same recorded
   source weights and seed. Do not compare a short RAE fine-tune directly with
   the released, fully trained VAE checkpoint as a causal result.
5. Train the matched VAE and RAE runs with the same demonstrations, steps,
   optimizer, batch size, action loss, inference steps, and checkpoint rule.
   Run separate head-width and noise-shift ablations only after the base pair.

## Gates

- Reproduction: checkpoint hash, native three-camera RT, cuRobo, and complete
  released-checkpoint evaluation must be audited before claiming improvement.
- Data: freeze train/dev/test demonstration manifests and content hashes;
  inspect episode and scene overlap; fit statistics only on train.
  The bounded ten-task archive selection is frozen in `DATA_SELECTION.md`.
- Numerical: one batch with finite latents, actions and losses; check shapes,
  nonzero intended gradients, save/reload equality, and raw video/action loss
  at matched noise/timestep/dtype.
- Engineering: a bounded small-data overfit run is an implementation check,
  excluded from the final endpoint.
- Scientific: at least three matched training seeds, fixed task/scene pairs,
  per-task success and technical failure rates, and paired confidence intervals.
  A trained model is not evidence of a visual-representation benefit until this
  comparison is complete.

The stage-one decoder pilot was registered at
`/root/zifan/openwam-runs/rae_decoder_20261023_seed42/registration.json`.
It trained for 1,000 steps on train-only frames; the selected step-1,000
checkpoint reached dev normalized L1 0.12870. A single test read after
selection gave RGB MAE 0.02844 and PSNR 23.15 dB over 1,002 sampled frames.
These are reconstruction metrics, not robot-task improvement.
That first decoder used per-token-normalized features. A calibrated RAE
requires a separately trained decoder on the calibrated features.
The matched 1,000-step calibrated decoder reached dev L1 0.12991 and test
RGB MAE 0.02880 / PSNR 22.75 dB on the same 1,002-frame test sampling.
Calibration did not improve this pixel-reconstruction metric. Its effect on
WAM denoising and task success remains unmeasured; do not infer either from
decoder reconstruction.
Both stage-one decoders were trained on single frames. The current DINOv2
WAM encoder has `pixel_decode=False`; these pilots do not yet reconstruct a
9-frame generated clip from its 3 temporally pooled latent frames. A temporal
decoder is required before reporting generated-video quality.

The released RoboTwin-Full checkpoint was trained on RoboTwin and may contain
the ten selected tasks or episodes. It is therefore the **evaluation
reproduction target only**, not the donor for this controlled comparison.
The matched VAE/RAE pilot will instead use the released
`OpenWAM-Alpha-Pretrain-Foundation-Model` checkpoint, revision
`52df4e66c82c5c8b480adcc8d01f4db7415dfb56`, weight SHA256
`180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d`.
The released pretraining mixture includes simulation data; episode-level
membership has not been audited, so this choice removes the known downstream
RoboTwin fine-tune overlap but does not prove zero pretraining overlap.
The RAE graft retains every shared trunk tensor and the action head by
name and shape, removes the native VAE, loads frozen DINOv2, and resets the
two visual projections. The VAE control resets those same projections with
the same per-run RNG seed. Width still differs (48 versus 768 channels), so
projection capacity is a separate required ablation before attributing gains
to representation quality. Register each WAM GPU run separately.

This host has CUDA A100s but currently cannot initialize SAPIEN's required
Vulkan RT renderer. Native closed-loop reproduction and the final paired
success-rate evaluation remain gated on a working RT host.

The matched single-seed pilots are registered under
`/root/zifan/openwam-runs/wam_pilot_20261023`. A detached
`research/rae/continue_pilot.py` process waits for the running RAE training,
verifies its final checkpoint, releases only the SHA-verified reproducible
graft weight file for disk space, and trains the VAE control. It then uses
`eval_teacher_forced.py` on the same 100 dev windows with four noise draws
each and `compare_teacher_forced.py` for per-task paired action-loss deltas.
The state file is `continuation_state.json`; any failure stops the sequence
and preserves logs. The video losses have different latent targets and are
not compared across variants.

### Completed single-seed pilot (2026-10-24)

Both 1,000-step jobs completed and wrote final checkpoints. The RAE checkpoint
SHA256 is `9005b330d857ff01cb5d11712e5ba593b72afc3ce600e51f0279acc0bcf8a739`;
the VAE control SHA256 is
`6aca452aa6f06ed358d685fb220e1a245b00fd422e54bb80436481db6e97b531`.
On the same 100 episode-disjoint dev windows (10 per task, four noise repeats),
mean teacher-forced **action** loss across tasks was 0.0239227 for RAE and
0.0201182 for VAE, so RAE was 0.00380449 higher (18.91%; lower is better).
RAE was lower on one of ten tasks (`open_microwave`) and higher on nine.
The full paired report is
`/root/zifan/openwam-runs/wam_pilot_20261023/paired_dev_comparison.json`.
This single-seed offline proxy does not establish a task-success difference.
The A100 host still cannot run the native NVIDIA Vulkan RT closed-loop simulator.

After model selection on dev, an identical 100-window, four-noise-repeat
teacher-forced evaluation was run on test episodes not used for WAM
training or checkpoint selection.
`eval_teacher_forced.py --split test` reuses the frozen training statistics;
`continue_test_eval.sh` ran VAE and RAE sequentially on GPU 4 and wrote
`paired_test_comparison.json` under the pilot run root. The test split is read
once for this predeclared comparison, and action loss remains an offline proxy.

The test comparison completed: on 100 test windows (10 per task, four fixed
noise draws per window), VAE action loss was 0.0183711 and RAE was 0.0235261,
a RAE increase of 0.0051550 (28.06%; lower is better). RAE was lower only on
`open_microwave`; it was higher on the other nine tasks. A hierarchical paired
bootstrap that resampled tasks and then windows gave a descriptive 95% interval
of [0.0018641, 0.0085579] for RAE minus VAE action loss (20,000 draws,
seed 20261024). The raw pair and uncertainty are in
`paired_test_comparison.json` and `paired_test_uncertainty.json`. These metrics
still do not establish a closed-loop success-rate difference.

### Software-rendered closed-loop pilot (2026-10-24)

An isolated Mesa Lavapipe 26.2.4 environment at
`/root/zifan/openwam-envs/lavapipe-rt` now runs SAPIEN's RT shader on CPU.
The RoboTwin OIDN path requires Vulkan/CUDA device interop, so
`ROBOTWIN_SOFTWARE_RT=1` replaces OIDN with no denoiser. The original 32
samples/pixel and path depth 8 work but were too slow for a task sweep.
A 1-sample test produced unusably noisy images; the paired exploratory sweep
uses 8 samples/pixel, path depth 4, one full-length `demo_clean` episode per
task, and the same initial evaluation seed for VAE and RAE. This renderer has a visible
domain shift from the released benchmark. Results from it must not be called
the native released-checkpoint reproduction or the official success rate.

Both matched checkpoints loaded into the policy server and completed a real
20-step `open_microwave` simulation smoke test at 8 samples/pixel, each with
0/1 success. The task's real limit is 1,500 steps, so these smokes only prove
that the policy, camera, cuRobo, and simulator loop runs. The paired ten-task
sweep was first registered in `research/rae/run_robotwin_software_rt.sh`; logs and
per-task status for the aborted 8-sample sweep are under
`/root/zifan/openwam-runs/robotwin_rt_eval_20261024/paired_8spp`.

The CPU Open Image Denoise 2.5.1 color-only RT filter provides a faster,
clearer exploratory configuration: one RT sample/pixel, path depth 2, then
LDR/sRGB CPU denoising in `Camera.get_rgb`. On the same initial
`open_microwave` scene frame, its PSNR against the 32-sample, no-denoiser
reference was 26.06 dB versus 26.32 dB for the 8-sample, no-denoiser frame.
Median post-warmup denoise time was 11.8 ms/frame. These are single-scene
diagnostics, not a guarantee of policy equivalence. The native renderer
denoises before display conversion and may use auxiliary render buffers;
this CPU bridge only uses the display RGB. Evidence and source frame hashes
are in `/root/zifan/openwam-runs/robotwin_rt_eval_20261024/oidn_image_probe.json`.
The 8-sample run was stopped mid-first task to avoid mixing render domains.
The first paired one-sample CPU-OIDN sweep used
`research/rae/run_robotwin_cpu_oidn.sh` and wrote to
`/root/zifan/openwam-runs/robotwin_rt_eval_20261024/paired_1spp_cpu_oidn`.

Mesa Lavapipe 26.2.4 proved unstable across tasks: `hanging_mug`,
`open_microwave`, and `place_bread_skillet` exited 139 with kernel faults at
the same `vk_meta_get_pipeline_layout` address in `libvulkan_lvp.so`. These
are technical failures, not policy failures. An isolated Mesa 26.2.5 install
at `/root/zifan/openwam-envs/lavapipe-rt-2625` completed 100-step repeats of
`hanging_mug` and `open_microwave` at the same initial scene seed and policy;
26.2.4 had crashed at steps 51 and 3 respectively. The completed paired sweep
used `research/rae/run_robotwin_cpu_oidn_mesa2625.sh` and wrote to
`/root/zifan/openwam-runs/robotwin_rt_eval_20261024/paired_1spp_cpu_oidn_mesa2625`.
Each task records the number of process attempts; one technical retry is
allowed, and failed attempts retain their logs. The released-checkpoint
sanity sweep was started by `research/rae/continue_official_after_paired_mesa2625.sh`
after all paired task attempts finished. The paired summary records 6/20
completed episodes, 14 terminal technical renderer failures, and two
matched-scene VAE/RAE task pairs with 0/2 successes for each model. It
reports terminal technical failures separately from policy success. The
sample is too small and too far from the native render domain for a
success-rate claim.
The 26.2.5 build also crashed in a first full-length `beat_block_hammer`
attempt at step 97 and a first `hanging_mug` attempt at step 30. A retry
completed all 400 `beat_block_hammer` steps, and the `hanging_mug` retry
continued beyond the first fault. The renderer is therefore still being
treated as failure-prone; a 100-step smoke is not a stability guarantee.
An isolated CPU raster diagnostic entered a real RoboTwin task but ran at
roughly 10 seconds/action in its early steps. Disabling Lavapipe threading
with `LP_NUM_THREADS=0` was several times slower than 16 threads and still
segfaulted at step 84/100 of a `beat_block_hammer` diagnostic. This rules out
thread count alone as a sufficient stability fix.
Mesa 26.1.6 was tested in a separate full-length `beat_block_hammer` process,
with a separate policy server and GPU pair. It also segfaulted, at step
354/400. The observed CPU Vulkan failures span 26.1.6, 26.2.4, and 26.2.5;
short smokes and a single completed retry do not establish simulator
reliability. The exploratory sweep recorded technical errors explicitly;
the held-out offline WAM test comparison was completed independently of the
simulator.

Two further matched WAM seeds (43 and 44) are prepared in
`research/rae/run_additional_seeds.py`. The runner waits for the registered
released-checkpoint controller to exit and checks that no RoboTwin evaluation
process still owns GPUs. It verifies each graft and
final checkpoint by SHA256, then evaluates both variants on the frozen dev and
test windows. The local overlay has under 15 GB free, so the final training
checkpoints are written to the shared NFS under
`/data/zifan-openwam-rae/wam_multiseed_20261024`; seed grafts temporarily use
`/dev/shm`. Logs, metrics, and hashes are written persistently under
`/root/zifan/openwam-runs/wam_multiseed_20261024`. The two additional seeds
are not yet results and do not replace the native closed-loop success gate.
After all three seeds have paired offline reports, use
`research/rae/aggregate_multiseed.py` to verify identical task/window/noise
settings and report the task-balanced mean across seeds plus a descriptive
bootstrap over seeds, tasks, and paired windows.
`research/rae/continue_multiseed_summary.py` waits for the training controller
to finish and writes the aggregate JSON files and a concise `SUMMARY.md` under
the persistent multi-seed run directory only if both new seeds completed.

For the visual-head width question, collect frozen DINOv2 train-split features
from a deterministic 8,192-window subset and fit a frozen 768-to-48-channel
S-VAE reducer. The collector now records the selected-index SHA256 and keeps
the full-dataset traversal as its default. A 48-channel DINO/S-VAE WAM is a
separate ablation against the 48-channel native VAE: its reducer is an added
learned component, so any difference cannot be assigned solely to head width.
The single-GPU collector writes its shards under
`/data/zifan-openwam-rae/svae_features`; `continue_svae48.py` verifies the
8,192-clip, 11,796,480-token archive and then trains a 1,000-step S-VAE on
GPU 2. This stage-one reducer is only preparation for a later matched WAM
run and is excluded from task-success claims.
The reducer checkpoint was saved at
`/data/zifan-openwam-rae/svae48_runs/svae_20261024_182838/svae.pt`, SHA256
`76519a1dc5f8199c728d892c7551ceb59432eebd60113f55bc692889b26cd1fe`.
On 100 episode-disjoint dev windows, its posterior-mean standardized feature
reconstruction had MSE 0.18898, versus 0.99139 for zero reconstruction;
cosine loss was 0.08621. See `RESULTS.md`. The S-VAE input-statistics cast
was corrected before WAM training so bf16 host casts preserve the exact fp32
stats from the registered train feature archive.
`continue_svae48_wam.py` then waits for the three-seed controller to release
all eight GPUs, builds a seed-42 Foundation graft with the frozen reducer,
trains a 1,000-step WAM under the same seed-42 data and optimizer settings,
and compares its dev/test action loss to the existing 48-channel VAE control.
The extra reducer and its 8,192-window stage-one subset remain explicit
differences, so this is a width-matched ablation rather than a pure
representation-only causal comparison.

The post-training video-noise shift sensitivity check is complete on dev.
Both seed-42 checkpoints were trained at shift 5 and evaluated at shifts 1,
5, and 10 on the same 100 windows and four noise draws. RAE action loss was
higher at all three shifts; see `RESULTS.md` and the registered reports under
`/root/zifan/openwam-runs/wam_pilot_20261023/video_shift_sensitivity`.
This diagnostic does not replace training a model at a different shift or
measuring closed-loop success.

## Related primary work

- Zheng et al., *Diffusion Transformers with Representation Autoencoders*,
  arXiv:2510.11690; reference implementation `bytetriper/RAE` (MIT), inspected
  at `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`.
- Jiang et al., *Rethinking Representations for World-Action Modeling*,
  [arXiv:2609.38163](https://arxiv.org/html/2609.38163). Its feature
  calibration, temporal bottleneck, and action-grounded shaping are candidate
  **separate** ablations, not part of the first RAE/VAE comparison. The paper's
  full method uses DINOv3-L features from several layers, a trainable
  2-frame/2-by-2-patch bottleneck, and action-only gradients into that
  bottleneck. Our frozen DINOv2-Base and offline-trained S-VAE do not implement
  that gradient routing, so their outcomes cannot test ReWAM itself. The
  authors' [repository](https://github.com/hustvl/ReWAM) currently contains
  only a release placeholder; no reference training code is available to
  transplant directly.
