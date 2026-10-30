# Experiment progress audit — 2026-10-30

This is an audit of completed evidence and remaining registered work, not a
final experiment report. Local GPU use is prohibited and the Hestia evaluation
was stopped by the user. This audit ran JSON aggregation on CPU only.

## Correct comparison of the partial native results

The accepted 2026-10-28 shutdown snapshot contains 1,000 VAE episodes (32
successes) and 280 RAE episodes (zero successes). Comparing those aggregate
percentages directly confounds model differences with task coverage.

Restricting **both models to the same 280 accepted scenes** gives:

| Outcome | Scene pairs |
|---|---:|
| VAE success, RAE failure | 1 |
| RAE success, VAE failure | 0 |
| Both success | 0 |
| Both failure | 279 |

The matched subset covers 100 `beat_block_hammer`, 100 `hanging_mug`, and 80
`open_microwave` scenes. It is incomplete and not balanced across the original
ten-task benchmark. No full native RAE benefit is established. These values
were recomputed from the accepted snapshot; the stopped Hestia queue was not
restarted or refreshed in this audit.

## Available offline training-seed evidence

Seeds 42 and 43 have completed VAE/RAE pairs and matching dev/test comparison
files. Their original per-window reports have been copied into
[`results/offline_evidence_20261030/`](results/offline_evidence_20261030/).
The existing aggregator checked matching tasks, window IDs, normalization and
split hashes, noise settings, and independent checkpoint identities.

| Split | Completed paired seeds | VAE action loss | RAE action loss | Relative RAE change |
|---|---|---:|---:|---:|
| Dev | 42, 43 | 0.0227026 | 0.0239145 | +5.34% |
| Test | 42, 43 | 0.0205905 | 0.0234757 | +14.01% |

These are teacher-forced flow-matching losses; lower is better. The
descriptive hierarchical bootstrap intervals for RAE minus VAE are
[-0.0037844, +0.0061165] on dev and [-0.0016566, +0.0073318] on test, using
20,000 draws with the registered bootstrap seed. Both intervals cross zero.
This is a **two-seed partial aggregate**, not the registered three-seed
endpoint, and it does not establish a closed-loop effect.

Seed 44 has an RAE checkpoint and completed dev/test reports, with action
losses 0.0221420 and 0.0209799 respectively. Its matched VAE checkpoint/report
is missing. The seed-44 controller state still says training started; the
checkpoint and evaluation artifacts demonstrate later progress, so that
controller label must not be treated as current experimental truth.

## Coverage of the original experiment plan

Five step-1,000 WAM checkpoints are present: seed-42 VAE/RAE, seed-43 VAE/RAE,
and seed-44 RAE. Their presence and byte sizes were checked. This audit did
not rehash the large weight files; recorded hashes come from existing
evaluation reports and the accepted native snapshot.

The registered native core is six matched-model evaluations of 1,000 episodes
each, two additional S-VAE48 evaluations of 1,000 episodes each, and the
official model on 50 tasks × 100 episodes × two modes: **18,000 distinct
core episodes**. The optional online-action evaluation adds 1,000 episodes.
The seed-42 VAE control reused in the ablation is counted once.

Only 1,280 core episodes were accepted at shutdown, leaving **16,720 core
episodes**, plus the optional 1,000, uncompleted. Required remaining work is:

- Seed-44 VAE training, offline evaluations, and full three-seed aggregation.
- Reconstruction S-VAE48 WAM training and offline evaluations.
- Action-informed S-VAE48/relevant WAM training and offline evaluations.
- Optional online action adapter training and its offline/native evaluations.
- Remaining seed-42 RAE native batches and native evaluation of the other
  matched models and the S-VAE48 ablations.
- Complete official-checkpoint evaluation in both native modes.
- Final paired/ablation/official summaries, uncertainty, and provenance checks.

Decoder reconstruction and reducer reconstruction reports already exist, but
their metrics cannot substitute for any of those policy evaluations. The
registered full-result builders remain incomplete and must continue to reject
missing work. A proposed small-model experiment would be a separately labeled
follow-up; it does not complete or replace the original registered plan.

## Why the execution order must change when compute is available

The plan already specified reproduction and bounded overfit gates, but the
execution expanded short WAM training to additional seeds before establishing
a working released-checkpoint native baseline and a reliable small-data
closed-loop implementation check. Offline loss was used as an interim
indicator despite its limited connection to rollout success. Neither more
short-run seeds nor a fixed epoch count resolves that missing evidence.

Before resuming expensive experiments on authorized resources, establish the
official native baseline and investigate concrete trajectory failures. A
small frozen-feature policy/world-model comparison can then test representation
utility at a measured budget. Choosing that additional experiment does not
turn the unfinished registered results into a completed reproduction.

Machine-readable audit and evidence fingerprints:
[`results/progress_audit_20261030.json`](results/progress_audit_20261030.json).
Shutdown and recovery details remain in
[`HANDOFF_20261028.md`](HANDOFF_20261028.md).
