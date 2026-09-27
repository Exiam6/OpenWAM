# Goal auxiliary confirmation: incomplete at the original collection deadline

Recorded 2026-09-22T19:21:18.931171-05:00. The collection deadline was 2026-09-22T19:15:55.548938-05:00; the
overall study deadline remains 2026-09-23T01:15:12-05:00. Neither was extended.

## Terminal result

| Task | Accepted | Required |
| --- | ---: | ---: |
| adjust_bottle | 20 | 20 |
| handover_block | 20 | 20 |
| place_object_basket | 16 | 20 |
| Total | 56 | 60 |

70 distinct candidates were started: 69 completed (56 accepted, 3 unstable,
10 expert-infeasible), and basket seed 802024 was interrupted at the original
collection deadline. The worker exit was 124. Its parent exited 1 at
2026-09-22T19:15:56.641086-05:00. The original generic infrastructure-failure
record is retained unchanged; this terminal assessment identifies the specific
budget timeout from the launcher deadline, worker wait bound, exit code and
terminal timestamps. The interrupted seed was not counted as expert-infeasible.

No goalaux worker or tmux job remained at audit. The assigned GPU was released
(1 MiB, 0% utilization, zero uncorrected volatile ECC in the sampled check).
No completed candidate was replayed after the recorded repairs; the interrupted
candidate will not be retried under this expired collection budget.

## Integrity and what remains unmeasured

The terminal archival audit reread all 56 accepted HDF5 hashes, expert plan and
replay success records, actual Curobo planning provenance and rendering device,
consecutive candidate order, per-task attempt cap, and completion before the
original deadline. All checks passed; accepted HDF5 hashes are distinct within
this incomplete cohort. This does NOT substitute for the unexecuted cross-cohort
identity gate against the previous 210 scenes or the prediction parity gate.
No fresh model outputs, target labels, scores, confidence intervals or acceptance
decisions were computed in this audit.

All six matched training runs and training-only readout fits/audits previously
completed. Both waiting scoring jobs stopped earlier at their dependency gates.
No confirmation-v3 was launched and no partial-cohort score will be substituted.
Under the frozen protocol, fewer than 20 accepted scenes for any task means
confirmation incomplete. Auxiliary supervision is neither validated nor falsified
by this collection timeout. Training success and identical latent dimensions do
not establish released-policy compatibility or closed-loop benefit.

The earlier K4-vs-L12 layer result remains separate: its noisy-state primary
criterion failed. Its initial-waypoint improvement was secondary and measured on
the same 60 old scenes; it cannot rescue that primary result or confirm the new
auxiliary intervention. This study is not a complete RAE-versus-pixel-VAE test.

## Retained evidence and next step

Raw terminal metadata/logs: fresh-v4-terminal/ (including original timeout and
all continuation worker records). Terminal audit: collection-terminal-audit.json.
Runtime HDF5 files remain in place; no datasets/checkpoints were added to Git.
All earlier path/null-grasp failures and amendments remain available.

Frozen scientific JSON SHA256: ac87b2067ad7e66f372fddfdda31818abcc1b3ef5e9f0ae70c92de672e6fc0a6
Frozen scientific Markdown SHA256: 1372d41f026fc3f3d1b32f39431799aad90244e821b6849d6ba2421c7a83ab0d

The goalaux confirmation branch is terminal/incomplete: retain all 56 accepted scenes, 70 distinct started candidates, six trained models and frozen readouts. Do not restart collection, score the partial cohort, refit, extend the original bounds or reinterpret incompleteness as benefit/no benefit. Scientific uncertainty remains whether fixed goal supervision jointly improves noisy initial-waypoint information and preserves state information; this run cannot answer it. The completed layer-study evidence and opt-in provenance patch are available for review in representation-evidence.md. Subsequent selfchecks should only verify unexpected changes or address the existing publication blocker; no duplicate GPU job/timer/agent. A future scientific experiment would need its own prospective protocol and cannot silently complete this expired cohort. Original overall study deadline remains 2026-09-23T01:15:12-05:00.

## Log transport note (2026-09-22 19:30 selfcheck)

The initial text-based transport normalized carriage returns in seven progress-bar
worker logs. All 42 other archived metadata/log files matched the remote byte
hashes. The seven remote logs have timestamps before collection ended, and their
normalized text matches the initial local copies exactly. Byte-exact originals
are now additionally preserved; the earlier text copies remain available.
This changes no HDF5 data, outcomes, model artifacts, scores or deadlines.
Private terminal-archive-index.json identifies the canonical byte-exact copy of
all 49 files for subsequent checks. The archive correction is not a new scientific
experiment or a rerun of the accepted-file integrity audit.
