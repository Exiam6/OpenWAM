# Self-check of ex3 against research_lab/TOP_PRIORITY (2026-10-17; first pass before results, updated 07:40 CDT after scoring)

Checked by the agent running the study; every row names the evidence file. "gap" rows are open items, not excuses.

## EXPERIMENT_PROTOCOL.md

| rule | status | evidence / gap |
| --- | --- | --- |
| 1 one branch, one hypothesis | pass (2×2 design, each contrast single-variable) | `PLAN_ex3_20261017.md` §Hypotheses; ex3-a and ex3-b are separate runs with separate claims |
| 2 written before run, not edited after | pass | PLAN committed 07f9b44f before the first claim (claims start 02:08 CDT); amendments listed in the PLAN, all before launch |
| 6–7 testable question, what it cannot answer | pass | PLAN §Question |
| 8 plan file contents | pass | PLAN has baseline, hypotheses + competing explanations, intervention diff, held fixed, metrics, prediction, criterion, budget/deadline, stop/rollback, outputs |
| 9 metric and test set fixed; no checkpoint selection | pass | final checkpoint only (`train.py`); fresh scenes frozen by expert screening before any policy loads (`collect_fresh.py`) |
| 10 task metric, not loss | pass | endpoint = native success; sigma-MSE labelled diagnostic only |
| 12 one variable at a time | pass after amendment | the 2026-10-10 primary (two variables) demoted to a labelled confounded contrast; `score.py` CONTRASTS |
| 13 paired, ≥3 seeds, each seed reported | pass | 3 seeds, paired scenes, `by_training_seed` in `score.py` |
| 14 control that separates explanations | partial | shift vs head separated by the 2×2; the Wan-VAE control arm (RAE's actual claim) is still missing — waiting node on the page |
| 15 benchmark version | pass | same simulator, assets, scene rule; renderer (OIDN 2.3.3) identical for every arm |
| 16 cost estimated per stage | pass | PLAN §Budget; measured 0.27 s/step → ~1.5 h per training |
| L0–L4 staged scale-up | pass / running | L0 config diffs (PLAN); L1 preflight on gpu-01 `records/rae-factorial-gpu01-preflight-*.json`; L3 renderer smoke on this host (scratch log) + per-cell simulator gate; L4 running |
| 17 every run recoverable | pass | per-run directory with config.yaml, batches.npy, initialization.json, metrics.jsonl, resume.pt; code-freeze.json hashes; `resource-amendment-gpu01.json` records host/GPU map |
| 18 no mid-run edits | pass so far | frozen manifests re-checked by every worker before each claim |
| 19 shared machine: cap, record host, never touch others | pass | 4 threads per job, host/GPU uuid recorded; no other user's process on gpu-01 at launch (all 8 GPUs at ~30 MiB) |
| 20 endpoint is the registered one | pass | `score.py` uses only final-checkpoint cells |
| 21 read model / objective / evaluation code | pass | `common.py`, `train.py`, `evaluate_policy.py`, `evaluate_cell.py`, `serve_policy.py` read; policy input = three cameras + 20-dim proprio + fixed prompt; success = native `env.eval_success`; held-out starts = fresh expert-screened scenes |
| 22 visualize actual policy input | pass, with gap | every policy call's three camera frames are logged and assembled into success / median / failure clips per arm and task (frame checked by eye: gripper at bottle). Gap: no external full-robot camera was logged, so the head camera is the closest scene view (REPORTING 13) |
| 23 cross-check against an independent source | pass | ex2 cell re-run twice on gpu-01: 8/20 and 10/20 vs gpu-02 11/20; the two gpu-01 runs disagree on 4/20 scenes, the hosts on 3/20, so the baseline reproduces within same-host noise (`records/ex3-cross-host-replication-run{1,2}-20261017.json`) |
| 24 two agents agreeing is not proof | noted | the cross-host replication is an independent machine, not a second reading of the same files |
| 27 scoped conclusion template | pass | `CONCLUSION_ex3_20261017.md` |
| 28 negative result diagnosed | pass | says what it rules out (effects beyond about ±15 pp on adjust_bottle) and what it cannot address (two zero tasks, VAE control, scale); post-hoc interval labelled as such |
| 29 rerun baseline after merge | n/a | nothing merged |
| 33 stop and ask | pass | stop conditions in PLAN; the deadline is enforced by every worker |

## EXPERIMENT_NAMING.md

| rule | status | evidence / gap |
| --- | --- | --- |
| format `ex{A}-{B}_{science}_{MMDD}` | pass | `ex3-a_shift16_1017`, `ex3-b_wideshift16_1017`; history table in the PLAN maps old study names |
| 1 register before launch | pass (plan file) / gap (sheet) | registered in the PLAN and on the page; the OpenWAM project has no Google Sheet yet (see below) |
| 3 only science in the name | pass | GPU ids and host live in the run directory and the amendment record |

## REPORTING.md and LOGIC_TREE.md

| rule | status | evidence / gap |
| --- | --- | --- |
| 1 conclusion first, plain language, numbers in tables | pass | page home view and chat reports |
| 2 chat in Chinese, pages in English | pass | |
| 3 every number from a file | pass | `build_ex3_page.py` reads run files and records; `sources.json` lists 24 files; ex1 numbers are copied from the dated report table and say so |
| 5 bad news stays visible | pass | ex0-shift retraction, ex2 negative wide-head reading and the two zero tasks are nodes on the tree |
| 6 test what you deliver | partial | video frame extracted and inspected; numbers traced to `success-summary.json`; the download button is not exercisable from this terminal |
| 9–13 video first, paired, policy-input view | in progress | clips assembled from logged frames once cells finish; three-camera policy-input view; success / median / failure per arm and task |
| 17 pages generated by a script | pass | `report-sources/ex3-logic-tree-20261017/build_ex3_page.py` + research_lab builder; never hand-edited |
| 18 download buttons | pass | every video and data file gets a button; bulk bundle to be built at the end |
| 19 one table, one protocol | pass | fresh-scene table holds only ex3 cells; ex2 numbers are shown as "earlier reading" in a separate column, never mixed |
| 22 handed-off work is checked | pass | worker PIDs verified after launch; background waits report each stage |
| LOGIC_TREE 1–9 | pass | root question, sub-questions, baseline node, one node per hypothesis, waiting nodes, draggable SVG tree, timeline from claim timestamps, home view counted (0 CJK), hash-routed single HTML |

## GOOGLE_SHEET.md / CREDENTIALS.md

Gap: the OpenWAM project has no live sheet and no Apps Script web app. Setting one up needs the user to deploy the template web app (`templates/google_sheet_webapp/`) and store its URL in `credentials/`; until then the registry lives in the PLAN and the page. No credential is used or printed anywhere in this study.

## Issues found while running (each recorded as an amendment or record, none after a policy result)

| time (CDT) | issue | action |
| --- | --- | --- |
| 02:45 | scene screening waited for training although it loads no policy | amendment 2: screening may overlap training |
| 02:47 | an unstable initial scene (RoboTwin UnStableError) stopped a whole task | amendment 3: skipped per RoboTwin's own rule; failed attempt archived |
| 02:55 | one evaluation cell per GPU left about half of each GPU unused | amendment 4: a second worker slot per GPU (scheduling only) |
| 03:05 | an expert-planner exception stopped place_object_basket screening | amendment 5: skipped per RoboTwin collect_data rule; task re-screened from its first candidate |
| 03:29 | the re-screen changed 5 / 24 expert verdicts: the expert planner is not deterministic | recorded; frozen cohort is the second screen, fixed before any policy was loaded |
| 03:40 | same-host repeat: episodes are not deterministic with fixed seeds (4 / 20 flips) | recorded; stated in the conclusion as a measurement caveat |
