# Layer study execution freeze — 2026-09-22

The12-hour user-approved plan starts13:15:12CDT and ends01:15:12CDT next day.
Primary contrast K4-SVAE48 vs L12-SVAE48; L6 andPCA are mechanism controls.
Prior80closed-loop trials and all completed diagnostics remain untouched.

First-hour interface preflight passed on native DINO-B/16: exactly equal K1,
no conditioning-frame dependence on future frames for all3sources, all150raw
episodes structurally complete. Source commitb6d8f83, actual result retained.

Extraction: existing seed42 episode permutation; first35train, next5development,
last10remain unused in this run (previously exposed; NOT independent test).
All3tasks,12linspace starts per episode over[0,n-33],9video frames every4raw rows.
Head-only resizedBILINEAR320Wx384H, exactly the planned single-camera comparison.
This differs from published policy's multiview input: future equal-budget policy
adaptation must use head-only in BOTH arms and report that scope explicitly.
Native encoder preprocessing and cv2 image decode follow the existing reader.
Each cached sample includes task/episode/frame/split and encoded-image hashes.

Layer numbers1-based. Capture block6,9,10,11,12 before model final norm; use
same learned final norm ONCE per selected block. L12 directly takes native
last_hidden_state. K4 averages normalized9-12 inFP32 then castsBF16. All share
native causal temporal pool. No output feature norm before reducer training.
Frozen encoder checkpoint SHA in interface-preflight.json. Same backbone in
all arms; no learned layer weights or source selection after outcomes.

PCA channel mean/std/covariance from training clips only, unwhitened48projected
coordinates, common outputLayerNorm before control readout. SVAE native config:
input768 latent48 heads16 layers3 intermediate2048 dropout0. Shared3-task model
per source/seed42,43,44. Batch3,one clip/task each update;2000steps. AdamWlr1e-4,
betas(.9,.99),weight_decay1e-4;warmup20,cosinefactor(.95+.05cos(pi*step/2000));
KLbeta1e-4 ramp400steps,cosineweight1,clipgrad1,bf16autocast,TF32disabled.
The identical per-seed task-balanced sampling schedule is reused across sources.
Development reconstruction every200steps is diagnostic only; finalstep2000saved.
No new control-aware loss, noise training, tuning or best-checkpoint selection.

Fresh data: collect_fresh.py frozen761f53d. Task seeds600000/601000/602000,
first20expert-feasible complete episodes within40consecutive candidates each.
Native expert planning then saved-path replay, full3camera records+labels,
all attempts retained. No model scoring. Individual subprocess<=600s,total<=2h;
infrastructure failure stops the collector, no automaticretry. Each accepted
file has SHA256. Planner fallback flag uses the same verified environment
compatibility path as earlier diagnostics; report exact backend from logs.

Readout and confirmation follow the full plan; their runnable code must be
checked/frozen before scoring. Existing development scores do not become fresh
confirmation. Not yet a fully validated OpenWAM policy adaptation implementation.
Current native profile blocked on missingDeepSpeed and idle same-nodepair; neither
has been counted as a successful training throughput measurement.

Native throughput preflight (before any policy comparison): dependencies installed
in a study-local target; no shared environment edits. Native published model with
original reducer used only for geometry/throughput profiling, not a performance
comparison. Head-only105training episodes, original published action normalization,
2GPU,batch1each,gradaccum4=globalbatch8,CPUoptimizeroffload,ZeRO2,bf16,
280microsteps=70optimizersteps. First80microsteps are20optimizerwarmup; remaining
200microsteps are50optimizermeasurements. Instrument native log_step without
changing losses; no checkpoints saved. Native max_steps counts MICROsteps, so a
future1000optimizerstep adaptation would require4000microsteps at this setting.
20min process-group timeout, only its own children; failure means unmeasured,
not zero throughput or a scientific negative. Timing via native steps_per_sec.
