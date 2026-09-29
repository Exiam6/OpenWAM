# Greene deployment — OpenWAM research branch

Target machine adaptation of `research/progress-20260927` for the **NYU Greene**
Slurm cluster (L40S sm_89 / H200 sm_90). Written 2026-09-28.

Per `research/migration/README.zh-CN.md` step 3, this is a **separate deployment
directory**. Nothing under `research/experiments/` or `research/migration/` is
modified; all frozen protocols, hashes and deadlines stay as they are.

## Status

| Component | State |
|---|---|
| Repo clone (`research/progress-20260927`) | `/scratch/zz4330/OpenWAM` @ `3b9b2e1` |
| Policy env | **built** — `/scratch/zz4330/conda/envs/openwam-policy`, py3.10.21, torch 2.7.0+cu126 |
| Benchmark env | **built** — `/scratch/zz4330/conda/envs/openwam-bench`, py3.10.21, torch 2.4.1+cu124, sapien 3.0.0b1, mplib 0.2.1, warp 1.13.0 |
| SAPIEN/mplib patches | **verified byte-identical to the frozen `after_sha256`** (see below) |
| RoboTwin @ `0aeea2d` | cloned |
| cuRobo @ `d64c4b0` (v0.7.8) | cloned; first build **failed** (job 18756828: env nvcc was 13.3, torch is cu124). Rebuild queued against a pinned CUDA 12.4.1 prefix (`owam-cuda124` → `owam-curobo`) |
| RoboTwin 2.0 assets | download queued (`owam-dl`) |
| Released `OpenWAM-Alpha-Sim-RoboTwin-Full` (24.8 GB) | download queued (`owam-dl`) |
| Private checkpoints / sealed scenes (68.24 GiB) | **not transferred**. GPU02 is on a UIUC private IP (172.22.224.85), unreachable from Greene, so a direct pull is impossible. Use the byte-exact delta path in `delta/README.md` (~4–5 GB relayed instead of 73 GB) |

## Environment equivalence evidence

The two documented upstream patches reproduce **exactly** on Greene. Both
pre-patch files matched `benchmark-setup.json`'s `before_sha256`, and after
applying RoboTwin `script/_install.sh`'s two `sed` edits both matched
`after_sha256` byte-for-byte:

| File | after_sha256 | result |
|---|---|---|
| `sapien/wrapper/urdf_loader.py` | `13aad693…a588059` | MATCH |
| `mplib/planner.py` | `23c51d08…fcf6c2cf3c` | MATCH |

This is the strongest cross-environment equivalence result available so far.
It covers the two patched files only — it does **not** establish inference or
render equivalence, which remains unproven (as on GPU02).

### Deliberate deviations from the frozen source machine

| Frozen | Greene | Reason |
|---|---|---|
| `build_arch: 8.9` | `TORCH_CUDA_ARCH_LIST="8.9;9.0"` | L40S **is** sm_89 — identical to the frozen arch. sm_90 added so the same build serves H200. Build-time only; no numerical effect. |
| `max_compile_jobs: 2` | `MAX_JOBS=8` | Build-time parallelism only. |
| `nvcc: /usr/local/cuda-12.4/bin/nvcc` | `/scratch/zz4330/conda/envs/cuda-12.4` (conda, `nvidia/label/cuda-12.4.1` only) | Greene has no system CUDA 12.4. The bench env's `cuda-toolkit=12.4.1` metapackage pulled 13.3 components; a separate, version-locked prefix fixes that without touching the patch-verified env. |
| host compiler: not recorded | gcc/g++ 12 (conda-forge) | nvcc 12.4 requires gcc ≤ 13; the bench env carries gcc 15. |
| `pytorch3d_installed: false` | not installed | Unchanged. Clean RGB-only eval requests no point clouds. |
| systemd worker + GPU UUID lock | Slurm job + `CUDA_VISIBLE_DEVICES` | Greene allocates GPUs per job; the UUID/ECC/idle admission guards are replaced by Slurm's allocation. |

Greene's L40S being sm_89 matters: it is the closest-equivalence hardware
available for evaluation. H200 (sm_90) is the training-shaped resource.

## Cluster notes

- Account `torch_pr_222_courant`; partitions `l40s_courant`, `h200_courant`.
- Courant group shares a **48-GPU cap**. `QOSGrpGRES` in `squeue` = at the cap,
  backfills when a leg releases. Downloads therefore run on `cs` (CPU) so they
  do not queue behind it.
- Greene **login nodes kill large downloads/unzips** (RoboTwin `objects.zip`
  was cgroup-killed). Run that work through `sbatch`.
- `/scratch` quota 5 TB, was 82.7% used / 74% inodes at setup.

## Layout

```
/scratch/zz4330/OpenWAM/research/greene/
├── README.md                     this file
├── sbatch/                       Slurm launchers
├── scripts/                      env build, downloads, asset transfer
└── records/                      transfer + run records (JSON)

/scratch/zz4330/openwam-runtime/
├── assets-source/                private artifacts land here (68.24 GiB)
├── benchmarks/RoboTwin/          simulator + envs/curobo
├── handoff/                      files.txt / SHA256SUMS / restore-aliases.py
├── logs/  runs/  cache/  tmp/  setup/
```

## Private asset transfer (user-run)

> GPU02 cannot be reached from Greene. Use **`delta/README.md`** unless a
> UIUC `ProxyJump` host is configured. Both paths finish with the same
> `scripts/finalize_private_assets.sh` verification.

Source `zifanz4@shenlong-gpu-02.cs.illinois.edu:/home/zifanz4/openwam-runtime`.
363 files, 73.27 GB (68.24 GiB): six final temporal checkpoints with their
configs/tokenizers, plus the sealed scene cohort. Manifest-driven — never
recurses into `runtime/`, caches or GPU locks.

```bash
tmux new -s owam
bash /scratch/zz4330/OpenWAM/research/greene/scripts/fetch_private_assets.sh
```

Resumable; verifies every file against `SHA256SUMS` and writes
`records/asset-transfer.json`.

## Constraint carried over from the frozen protocol

The GPU02 worker holds the sealed cohort under the **unchanged
2026-09-29 17:48:49 CDT** deadline, and `lane.py` is a from-scratch finite
queue, **not** a cross-machine resume tool. Before Greene touches any temporal
evaluation cell, the split must be written down and confirmed so the two
machines never execute the same cell. Old-machine and Greene outcomes are not
merged or selected between.

The first Greene lane avoids this entirely: the **official-checkpoint reference
baseline** (`NEXT_STEPS` item 2 — released checkpoint, same 3 tasks × 20 clean
scenes) is explicitly unrun and needs no private asset.
