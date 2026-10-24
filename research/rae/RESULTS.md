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
the fault. A paired one-episode-per-task exploratory sweep is still running
with Mesa 26.2.5 and at most one technical retry per task. Current attempts,
successes, and terminal technical errors must be read from
`/root/zifan/openwam-runs/robotwin_rt_eval_20261024/paired_1spp_cpu_oidn_mesa2625/status.tsv`
and the task logs. An exit code of 139 is a simulator failure, not a robot-task
failure. The released RoboTwin-Full checkpoint sanity sweep is queued after
the paired sweep. No native released-benchmark reproduction or reliable
closed-loop RAE-versus-VAE success comparison has been established yet.
