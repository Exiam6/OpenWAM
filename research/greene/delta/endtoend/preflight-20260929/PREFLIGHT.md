# Endtoend transfer preflight — 2026-09-29

Status: STOPPED before packing, per ENDTOEND_TRANSFER.md section 3.
Tools revision: d88e6da (research/progress-20260927).
Source files were read only; the running GPU02 checkout and workers were untouched.
No new endtoend assets have been uploaded.

## Required guardrails failed

The documented command reports 6 frozen files + 2,572,461 added files,
2,163.80 GB total (limit approximately 120 GB). Repeated non-checkpoint
`runtime/assets/objects.zip` files are 3.738 GB each (limit 3 GB).
See dryrun-summary.txt for the exact summary and largest files.
All requested source paths exist: MISSING list is empty.

The detailed file listing exhausted the approximately 8 MB remaining on /tmp
and raised `OSError: [Errno 28] No space left on device`. The summary completed
before that error. The task's partial log was moved to
/home/zifanz4/endtoend-transfer-work/endtoend-dryrun.txt, restoring the space.

## Cause and required transfer corrections

pack_delta.py uses os.walk(..., followlinks=True). Scene runtime directories
link to the shared RoboTwin installation, including assets, data, and .git.
For example, endtoend-20260923/scenes/adjust_bottle/seed-1100000/runtime/assets
points to /data02/zifanz4/openwam-experiments/benchmarks/RoboTwin/assets.
Dereferencing these links repeats the shared runtime across the cohort.

A separate read-only inventory without following or including symlinks finds
84,008,688,246 bytes (84.009 GB) of regular files and 2,746 symlinks.
See no-follow-inventory.json. This is diagnostic only: blindly omitting the
links is not a complete transfer, since Greene must recreate their targets.
Define the portable link/recreation plan and rerun the size gate first.

The documented --add-tree assets/dinov3-study also skips the required
encoder.safetensors (237,763,424 bytes). Include it using --add-tree-all.

assets/wan22-vae-baseline/Wan2.2_VAE.pth is 2,818,839,170 bytes. The current
packer copies non-safetensors files directly into files/, so it would exceed
the plain-Git file limit. Large plain files need chunked transport with exact
reconstruction and checksum validation before upload.

/data02 has approximately 15 GB free. Use a staging filesystem with enough
space for both the pack and Git objects (the home filesystem has about 87 GB).
Disable Git binary delta searching for checkpoint parts (pack.window=0),
and push commits in batches below 2 GB, as in the preceding Wan48 transfer.

## Existing successful transfer

Wan48 is already verified on Greene: records/wan48-transfer.json reports all
272 files verified at pack commit 6976c2122d016ad96e2ab288926b2a893d4fee74.
Do not retransmit these files.
