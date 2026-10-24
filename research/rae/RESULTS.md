# OpenWAM RAE visual representation pilot results

All results below use the ten preselected RoboTwin `clean_50` tasks and the
same Foundation donor. The native VAE and frozen-DINOv2 RAE models were each
trained for 1,000 optimizer steps from the registered matched initialization.
This is one training seed per variant.

## Episode-disjoint offline evaluation

Action loss is a teacher-forced flow-matching proxy; lower is better. Every
row is the mean of ten fixed windows per task, with four fixed noise draws per
window. The train-only action normalization is reused for dev and test. Video
losses target different latent spaces and are not compared across models.

| Split | VAE action loss | RAE action loss | RAE − VAE | Relative change |
|---|---:|---:|---:|---:|
| Dev | 0.0201182 | 0.0239227 | +0.0038045 | +18.91% |
| Test | 0.0183711 | 0.0235261 | +0.0051550 | +28.06% |

RAE had lower action loss on `open_microwave` and higher loss on the other
nine tasks in both splits. A hierarchical paired bootstrap over the ten tasks
and their ten test windows gives a descriptive 95% interval of
**[+0.0018641, +0.0085579]** for RAE minus VAE test loss (20,000 draws,
seed 20261024). The interval reflects sampling across these selected tasks
and windows; it does not include training-seed variability. The per-window
records and checkpoint hashes are in `paired_dev_comparison.json`,
`paired_test_comparison.json`, and `paired_test_uncertainty.json` under
`/root/zifan/openwam-runs/wam_pilot_20261023`.

The training logs show that both action losses were still falling at step
1,000. These are means of the progress-bar action loss over the indicated 100
training steps, not held-out or paired-window metrics:

| Training steps | VAE action loss | RAE action loss |
|---|---:|---:|
| 1–100 | 0.088361 | 0.082933 |
| 901–1,000 | 0.014388 | 0.017144 |

RAE started lower on the first 100 stochastic training batches but ended
higher on the last 100, consistent with the held-out action-loss ranking.
The video losses in those logs have different targets and cannot be ranked
across the two variants.

### Post-training video-noise shift sensitivity

Both checkpoints were trained at video shift 5. Changing only the evaluation
video scheduler on the same 100 dev windows and four fixed noise draws per
window gave the following teacher-forced action losses:

| Evaluation shift | VAE | RAE | RAE − VAE |
|---:|---:|---:|---:|
| 1 | 0.0190375 | 0.0236975 | +0.0046601 |
| 5 (training setting) | 0.0201182 | 0.0239227 | +0.0038045 |
| 10 | 0.0209389 | 0.0241404 | +0.0032015 |

RAE remains worse on this offline metric at every tested shift. The gap
narrows at shift 10 because both variants' losses increase, so this does not
show a useful RAE improvement. These are fixed-checkpoint evaluations, not
models retrained at the alternate shifts or closed-loop results. Registration,
per-window values, and the comparison are under
`/root/zifan/openwam-runs/wam_pilot_20261023/video_shift_sensitivity`.

### 48-channel S-VAE reducer validation

A 768-to-48-channel DINOv2 S-VAE was trained for 1,000 steps on a fixed
8,192-window subset of the training split. On 100 episode-disjoint dev
windows (10 per task), deterministic posterior-mean reconstruction of its
standardized DINO features gave MSE **0.18898** and cosine loss **0.08621**.
A constant zero reconstruction in the same standardized space had MSE
**0.99139**. These values measure feature reconstruction only; they do not
measure WAM action prediction or robot success. The per-window report is
`/root/zifan/openwam-runs/svae48_dev_reconstruction.json`.

The reducer checkpoint retains full-precision training statistics when the
host model is cast to bf16. The earlier cast path rounded the statistics;
commit `d9e69ff` fixed this before the width-matched WAM training and an
exact-value check against the registered feature archive passed.
The bf16 forward still casts statistics to bf16 for arithmetic; the fix
preserves their stored fp32 values for later dtype changes. A ten-window
initialization smoke had identical action losses before and after the fix.

### Held-out test by task

Each row averages ten windows and four noise draws per window. The difference
is RAE minus VAE, so a positive value favors VAE on this offline metric.

| Task | VAE | RAE | Difference |
|---|---:|---:|---:|
| beat block hammer | 0.0148522 | 0.0233269 | +0.0084747 |
| hanging mug | 0.0171325 | 0.0202819 | +0.0031495 |
| open microwave | 0.0088642 | 0.0060661 | −0.0027981 |
| place bread skillet | 0.0239306 | 0.0338496 | +0.0099190 |
| place burger fries | 0.0105782 | 0.0181544 | +0.0075762 |
| place dual shoes | 0.0374516 | 0.0462277 | +0.0087761 |
| shake bottle horizontally | 0.0246101 | 0.0259697 | +0.0013595 |
| stack blocks three | 0.0152071 | 0.0219840 | +0.0067768 |
| stack bowls three | 0.0195595 | 0.0243716 | +0.0048120 |
| stack bowls two | 0.0115250 | 0.0150292 | +0.0035042 |

## Closed-loop status

The A100s expose CUDA but no NVIDIA Vulkan graphics device. CPU Mesa Lavapipe
with RT and post-render CPU OIDN can run real RoboTwin scenes, cameras,
cuRobo, and policy actions. Its images differ from the released renderer.
Long rollouts intermittently segfault in native rendering code on Mesa
26.1.6, 26.2.4, and 26.2.5; disabling Lavapipe threading did not eliminate
the fault. The paired one-episode-per-task exploratory sweep finished with
Mesa 26.2.5 and at most one technical retry per task. Six of 20 planned
episodes completed; 14 had terminal technical renderer failures. Of the
completed episodes, two VAE/RAE task pairs had matching accepted scene seeds
(`hanging_mug` and `place_dual_shoes`). Both policies failed both episodes,
so the exploratory paired result is 0/2 versus 0/2. This sample cannot
support a reliable success-rate comparison. Per-task attempts and outcomes
are in
`/root/zifan/openwam-runs/robotwin_rt_eval_20261024/paired_1spp_cpu_oidn_mesa2625/status.tsv`
and `summary.json`. RoboTwin may skip an initial scene seed when setup fails;
the summary includes a task in `pairs_complete` only when both checkpoints
finished and their accepted scene seeds match. An exit code of 139 is a
simulator failure, not a robot-task failure. The released RoboTwin-Full
checkpoint sanity sweep is now running. No native released-benchmark
reproduction or reliable closed-loop RAE-versus-VAE success comparison has
been established yet.
