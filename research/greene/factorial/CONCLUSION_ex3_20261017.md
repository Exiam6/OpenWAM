# ex3 conclusion — RAE noise shift and wide head on OpenWAM RoboTwin success (2026-10-17)

State: **the experiment finished and the registered conclusion holds as written below.** 900 / 900 planned attempts, 0 technical failures, scored by the registered `score.py` (records: `records/ex3-success-summary-20261017.json`).

## Result

Success counts on the 60 fresh expert-screened scenes (20 per task) × 3 training seeds, one host (shenlong-gpu-01), one renderer:

| arm | adjust_bottle | handover_block | place_object_basket | per seed 42 / 43 / 44 (of 60) |
| --- | --- | --- | --- | --- |
| raw768 native, shift 5 (baseline) | 29 / 60 (48.3 %) | 0 / 60 | 0 / 60 | 9 / 12 / 8 |
| raw768 native, shift 16.43 (ex3-a) | 31 / 60 (51.7 %) | 0 / 60 | 0 / 60 | 5 / 11 / 15 |
| raw768 wide head, shift 5 (ex2-b) | 26 / 60 (43.3 %) | 0 / 60 | 0 / 60 | 9 / 9 / 8 |
| raw768 wide head, shift 16.43 (ex3-b) | 29 / 60 (48.3 %) | 0 / 60 | 0 / 60 | 8 / 9 / 12 |
| S-VAE 48 (ex2-c) | 26 / 60 (43.3 %) | 0 / 60 | 0 / 60 | 11 / 9 / 6 |

Registered contrasts (task-balanced paired difference, scene-cluster bootstrap 95 %, 10,000 draws):

| contrast | single variable | Δ | 95 % interval | per seed (pp) | verdict |
| --- | --- | --- | --- | --- | --- |
| H1 | shift 5 → 16.43, native head | +1.1 pp | −3.3 … +5.6 | −6.7 / −1.7 / +11.7 | undecided |
| H2 | native → wide head, shift 5 | −1.7 pp | −5.6 … +1.7 | 0.0 / −5.0 / 0.0 | undecided |
| H3a | native → wide head, shift 16.43 | −1.1 pp | −4.4 … +1.7 | +5.0 / −3.3 / −5.0 | undecided |
| H3b | shift 5 → 16.43, wide head | +1.7 pp | −2.2 … +5.6 | −1.7 / 0.0 / +6.7 | undecided |
| diagonal (confounded, two variables) | wide head + shift 16.43 | 0.0 pp | −3.3 … +3.3 | −1.7 / −5.0 / +6.7 | undecided |
| S (secondary) | raw768 → S-VAE 48 | −1.7 pp | −7.2 … +4.4 | +3.3 / −5.0 / −3.3 | undecided |

## Scoped conclusion (EXPERIMENT_PROTOCOL rule 27 template)

Under 765 training demonstrations, 20,000 steps, a 12-layer 1024-wide DiT on raw 768-dim DINOv3 latents, and 60 fresh RoboTwin scenes on one host:

- **H1** changing the video-noise shift from 5 to 16.43 changed task-balanced success by +1.1 pp (per seed −6.7 / −1.7 / +11.7). Hypothesis: **undecided**; the prediction (|Δ| < 5 pp) held.
- **H2** the RAE wide head at shift 5 changed it by −1.7 pp (0.0 / −5.0 / 0.0). **Undecided**; the prediction (zero or negative) held. It replicates the direction of ex2 (24 vs 32 of 60 on old scenes) at a smaller size.
- **H3a / H3b** at the other factor's new level, the head changed it by −1.1 pp and the shift by +1.7 pp. **Undecided**; no interaction was detected. The two RAE changes together (confounded diagonal) gave exactly 0.0 pp.
- The cheap diagnostic moved as designed and did not transfer: the wide head lowers unweighted video MSE up to about 4× at σ ≥ 0.5 (all three seeds), and shift 16.43 trades about 3 % worse low-noise fit for slightly better high-noise fit — while closed-loop success did not move.

**Does not show:**
- Anything about handover_block and place_object_basket: every arm scored 0 / 60 on both (as in ex2). All information comes from adjust_bottle, so the task-balanced Δ is one third of the adjust_bottle Δ.
- Effects smaller than roughly ±15 pp on adjust_bottle. A post-hoc, unregistered adjust_bottle-only interval (`records/ex3-adjust-bottle-posthoc-20261017.json`) is −10 … +17 pp for H1 and −15 … +5 pp for H2; it states the detectable size, not a verdict.
- Whether RAE helps relative to a VAE latent (no Wan-VAE arm), or at official scale (118k steps, 30-layer DiT).
- Generalization over training randomness (three seeds; per-seed differences change sign for every contrast).

**Measurement caveats found during the run (self-check):**
- Episodes are not deterministic with every seed fixed: two gpu-01 runs of the same ex2 cell disagree on 4 / 20 scenes, and the expert planner also changed its verdict on 5 / 24 re-screened candidates. The paired design remains valid; intervals are wider than a deterministic evaluator would give.
- gpu-01 reproduces the ex2 gpu-02 baseline within that noise (8 / 20 and 10 / 20 vs 11 / 20); absolute rates are not compared across hosts.

**Next: stop trying** the shift and the wide head on this policy at this scale (the decision written in the PLAN for "undecided"). The two zero tasks, not representation details, bound what any arm can show; the open decisions for the user are (1) the missing Wan-VAE control arm, and (2) whether to spend compute on why handover_block and place_object_basket stay at zero (the official checkpoint reaches 20 / 20 and 18 / 20 there) before any further architecture comparison.
