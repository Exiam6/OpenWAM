# Endtoend dependency transfer to Greene

[02:57:39] DONE  source 84.03 GB -> pack 14.70 GB  (reused from reference 68.17 GB, deduplicated 1.16 GB)

- Source files: 4,379; source bytes: 84,030,142,327.
- Pack bytes before this README/authorization: 14,701,968,104.
- All six frozen checkpoint hashes matched; every checkpoint reuses >11 GB from Greene's reference.
- MISSING source paths: none.
- Wan48 checkpoints and training data already on Greene are not retransmitted.
- DINO encoder and shared tokenizer are included.
- All physical files are at most 95 MiB, including chunked VAE and compressor files.
- Shared RoboTwin trees are represented by 2,746 portable links, not duplicated.
- Source files were read only; no training or evaluation was launched.

## Retrieve and reconstruct

Repository: https://github.com/Exiam6/OpenWAM.git
Branch: transfer/endtoend-20260929
Use the verified full branch SHA supplied with the handoff as PACK_SHA.

```bash
sbatch --export=ALL,TRANSFER=endtoend,PACK_REPO=https://github.com/Exiam6/OpenWAM.git,PACK_BRANCH=transfer/endtoend-20260929,PACK_SHA=<FULL_SHA> \
  /scratch/zz4330/OpenWAM/research/greene/sbatch/delta_unpack.sbatch
```

After that job succeeds, restore aliases against Greene's existing RoboTwin:

```bash
python3 /scratch/zz4330/openwam-runtime/handoff-endtoend/restore-links.py \
  /scratch/zz4330/openwam-runtime/assets-endtoend \
  --robotwin /scratch/zz4330/openwam-runtime/benchmarks/RoboTwin
```

The link restorer refuses conflicting existing files and reports missing targets.
Resolve any missing target before evaluation. Checkpoint/config path adaptation
and deployment gates remain part of the Greene evaluation protocol.

## Transport fixes and verification

The patched packer is included at share-eval/pack_delta.py. It skips symlinks
with --no-follow-links and segments oversized plain files using empty header/tail
and blob segments, compatible with Greene's existing unpack_delta.py. A round-trip
test restored raw files, safetensors, and small files byte for byte; a symlink-cycle
test confirmed no recursive expansion. Link restoration was tested for idempotency
and protection of conflicting files. All internal alias targets are included.

The packing checklist is share-eval/ENDTOEND_TRANSFER.md from tools revision
d88e6da; the above corrections supersede its recursive tree-transfer examples.
PACK_SHA256SUMS covers manifest, pack parts, plain files, and handoff helpers.
