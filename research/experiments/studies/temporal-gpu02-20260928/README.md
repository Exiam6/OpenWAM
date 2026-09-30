# Latest GPU02 temporal progress and cluster ownership

Captured **2026-09-30T16:45:25.164932-05:00**. GPU02 worker active: **True**.
**882/1080 complete**, 16 technical failures,
2 unfinished/in-flight directories, 180 never attempted; zero duplicate identities.
Current: learned-seed44 / place_object_basket / head_camera_yaw_5deg.
Authorized resume deadline: **2026-10-01T18:43:29.998112-05:00**; the September 29 deadline below is historical.
Every 30 minutes, the GPU02 systemd monitor checks and resumes eligible pending work.

Torch/Greene must not launch this temporal study. See [handoff](../../../greene/GPU02_TEMPORAL_HANDOFF.md), [latest progress](snapshots/20260930T164525/progress.json), and [attempt ledger](snapshots/20260930T164525/attempt-ledger.json).

The following September 28 archive (including root progress.json and runtime-source) is historical; its deadline and official-baseline status are superseded by the latest snapshot and cluster records.

---

# GPU02 temporal evaluation and resume snapshot

Captured **2026-09-28 18:07:09 CDT**. This archive is a snapshot, not a live dashboard.

## Current state

- Expected:1080 paired cells =2routes x3training seeds x3tasks x3conditions x20scenes.
- Completed:280 (mean-seed42:177; learned-seed42:103). Infrastructure failures:3. In-flight/unfinished:1. Not started:796. Duplicate identities:0.
- Current:learned-seed42 / handover_block / head_camera_yaw_5deg. GPU02 physical GPU5, UUID GPU-a4c607bb-f8f8-b999-19ea-0eb65290bb10. Worker observed active.
- Deadline remains2026-09-29T17:48:49.941691-05:00. This archive does not extend it or authorize extra GPUs.

## Provenance and scope

The user reported the old machine experiments stopped and explicitly requested a new independent run. The GPU02 run began05:15 CDT. It reuses the six fixed checkpoints and sealed scene cohort; it is not a new independent cohort. Do not merge or select between old-machine and new-machine outcomes. Old live outputs were not reachable by SSH; the old218/1080 snapshot remains historical.

This is a custom native295M three-task temporal pooling ablation, not the official OpenWAM benchmark. It retains native RoboTwin task success, step limits, action conversion and WebSocket interface. It uses20 fixed scenes/task, clean/Gaussian.10/yaw5degree conditions, compile off and DiT cache off. The repository's published reference protocol uses released checkpoints,50tasks,100episodes/task, Clean/Randomized modes, compile/cache on. Official checkpoint and capacity/data/budget-matched original baselines remain unrun. A60episode three-task clean official-reference run was proposed, not launched. No policy-benefit or official benchmark reproduction claim follows from this partial run.

## Failure and resumption

The initial launcher stopped09:16 with123 completed outcomes at place_object_basket/clean scene1102008: initial-render MAE exceeded1. User then authorized resume and30minute monitoring. Resume-v1 started12:15 and skips every previously attempted scene identity, including incomplete/failed directories. It records render-mismatch MAE and actual image, then advances independent scenes; other scene subprocess failures are retained before advancing groups. Systemic/pre-scene failure stops for diagnosis. Completed and failed cells are not automatically replayed.

At this snapshot the same mean-seed42 scene1102008 failed under all three conditions. Gaussian and yaw initial head MAE were1.0315234375; the first clean failure did not save its numeric MAE. The threshold remains<=1. These are missing valid evaluations, never counted as policy task failures or silently dropped. All attempted is not equivalent to all valid.

## Environment migration

Exact transfer:363files,73,274,734,050bytes, all SHA256 verified,300scene aliases. Shared simulator assets were retained separately. PyTorch/CUDA moved to cu128 for Blackwell; benchmark torch2.4.1 became2.7.0; pinned cuRobo built forsm120. Original documented SAPIEN/mplib patches retained. Explicit renderer PCI pinning and copied robot YAML path remapping are operational adaptations. OIDN2.0.1 became2.3.3 because the old CUDA denoiser could not create a Blackwell device; originals retained. Development-only gates passed. Cross-environment inference/render equivalence has not been established. See migration/ records.

## Operations

Live root on GPU02:/home/zifanz4/openwam-runtime/temporal-new-20260928. Run `python3 <root>/progress.py` there; root status.json is the first stopped attempt, resume-v1/status.json is current. Runtime-source is an exact deployment archive with absolute paths, not a portable installer.

User services:openwam-gpu5-worker.service, openwam-gpu5-monitor.service; timer:openwam-gpu5-monitor.timer at minutes00 and30, plus worker termination inspection. User lingering enabled. The monitor checks inventory and starts pending work only when the reserved GPU is available; it does not kill other users' jobs. It stops at the original deadline or when all cells are attempted. Pause/disk/ECC/GPU-lock controls remain. Systemic failure is logged for diagnosis; this OS timer does not send chat messages or perform autonomous code repairs.

## Archived evidence

progress.json contains grouped counts, current worker and monitor snapshots. result-records.json contains every terminal result available at capture and its original file SHA256; traces and actual failure images remain on GPU02. runtime-source/ contains resume code, protocol and systemd units. Its code-manifest.json uses deployed absolute paths. archive-manifest.json hashes files as archived here; it does not alter prior frozen manifests. Migration JSON records are historical engineering evidence, not current live status.
