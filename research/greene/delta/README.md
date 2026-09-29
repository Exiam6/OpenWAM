# Byte-exact delta transfer (GPU02 → Greene without a direct link)

GPU02 (`shenlong-gpu-02`, 172.22.224.85) is on a UIUC private network and
unreachable from Greene, so `scripts/fetch_private_assets.sh` cannot pull.
Instead, a small pack is built on GPU02, carried by any relay (a laptop with
both VPNs, a cloud drive, …), and the original 363 files are rebuilt on
Greene **byte-for-byte**.

**Preferred relay: git.** See [`GPU02_GIT_TRANSFER.md`](GPU02_GIT_TRANSFER.md):
pack with `--part-mib 95`, push to a **private** repo with `git_push_pack.sh`,
then on Greene `sbatch --export=ALL,PACK_REPO=<repo> sbatch/delta_unpack.sbatch`.

## Why it is small

`save_checkpoint` (`openwam/model/architectures/base.py`) writes the whole
state dict, frozen modules included. In the released checkpoint the frozen
umT5 text encoder alone is 11.36 GB, and each private checkpoint is 12.19 GB.
If those frozen bytes equal the released checkpoint's, Greene already has
them. The six seeds also share their other frozen modules. Only the trained
DiT + action weights (~295M params) must travel.

Estimate before packing: ~4–5 GB instead of 73.27 GB. **The real number is
what `pack_delta.py` prints.** If the private text encoder does not match,
those tensors are simply packed and the result is larger, not wrong.

## Integrity

A safetensors file is an 8-byte length, a JSON header, then raw tensor bytes.
The pack stores each original header verbatim plus a per-tensor recipe, so
the rebuild is exact rather than a re-serialisation. Checks:

1. GPU02: each source file is re-hashed and must equal `SHA256SUMS`, or packing aborts.
2. Relay: `PACK_SHA256SUMS` is verified before anything is built.
3. Rebuild: every tensor range is re-hashed as it is written.
4. Every rebuilt file must equal its **original** sha256 from GPU02, or it is deleted.
5. `finalize_private_assets.sh` runs an independent `sha256sum -c SHA256SUMS`.

The pack contains no weights from the public checkpoint and nothing outside
`files.txt`. On GPU02 it only reads. It does not touch the worker, its
runtime directory, the sealed cohort's execution or its deadline.

## Steps

**1. Greene.** Build the reference index (a Slurm CPU job, already run):

```bash
sbatch research/greene/sbatch/delta_reference.sbatch
# -> /scratch/zz4330/openwam-runtime/delta/reference-index.json
```

**Relay to GPU02:** `reference-index.json` + `delta/pack_delta.py`.

**2. GPU02.** Standard-library Python only, low I/O priority next to the live worker:

```bash
nice -n 19 ionice -c3 python3 pack_delta.py \
    --root /home/zifanz4/openwam-runtime \
    --reference reference-index.json \
    --out /home/zifanz4/owam-delta
```

The output directory needs as much free space as the pack itself (see
`df -h /home/zifanz4`). The pack is split into ~1 GiB `pack/part-*.bin` files.

**Relay to Greene:** the entire `owam-delta/` directory →
`/scratch/zz4330/openwam-runtime/delta/pack/`.

**3. Greene.** Rebuild and finalize:

```bash
sbatch research/greene/sbatch/delta_unpack.sbatch
```

This writes `assets-source/` and `handoff/` exactly as the rsync path would,
then `records/asset-transfer.json`. It is safe to re-run.

## Reference

`OpenWAM-Alpha-Sim-RoboTwin-Full/checkpoint_step_118655.safetensors` (public,
24.81 GB, BF16). Its whole-file sha256 is stored in `reference-index.json`,
and `unpack_delta.py` refuses a pack built against a different index.
