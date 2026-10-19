> **2026-10-11 最新指令：暂停 GPU。** 用户要求“15号之前先不用gpu”。按 America/Chicago，2026-10-15 00:00 前不运行任何项目 GPU 任务，覆盖此前夜间空卡授权。gpu02 本轮全部 16 个训练/后续调度服务已停止并核实 inactive，无本轮残留进程；已写入 runtime `paused.json`，停止本轮监控 timer。本轮原 10-12 截止不延长，不在 15 日自动恢复。CPU/文档工作不受此限制。见 [暂停记录](records/rae-factorial-gpu-pause-20261011.json)。下方授权和运行状态为历史记录。

> **2026-10-11 夜间资源授权已部署：** 每天芝加哥时间 01:00–09:00 可使用 gpu02 任意空闲卡，其他时段仅 GPU1/2；本轮截止仍为 10-12 06:52:56 CDT。两项 GPU 预检已通过，正式训练等待空卡。每 30 分钟服务器监控覆盖八卡，详见 [最新计划与授权记录](EXPERIMENT_PLAN.md)。下方运行状态为历史快照。

> **2026-10-10 已获授权并启动等待服务：** 用户已允许 GPU1/2 空闲时运行、最长 36 小时。独立 study `rae-factorial-20261010` 的训练及后续评估服务均已启动；截至 19:07 CDT，GPU1/2 仍被其他任务占用，没有领取实验尝试。统一截止时间 **2026-10-12 06:52:56 CDT**，包含等待时间。见 [服务、PID 和哈希核验记录](records/rae-factorial-20261010-launch.json)。下方“暂不启动实验”属于授权前历史状态。

> **2026-10-10 RAE 结果更正：** 旧 shift 实验的约 1.7 倍 loss 差距主要来自训练权重缩放；CPU 去权重后比值 0.99685–1.02177。“shift 路线已关闭”的结论撤回。已新增固定 sigma、无权重的诊断工具；尚未运行模型，尚无 RoboTwin 成功率提升。见 [更正及证据](RAE_SHIFT_RESULT.md)、[下一轮草案](EXPERIMENT_PLAN.md#2026-10-10-rae-宽头与偏移联合验证草案未启动)。

> **两方是 UIUC Shenlong 与 NYU Torch；`gpu02` 是 Shenlong 节点的 SSH 别名，不是集群。**

> **2026-10-10 已验证贡献：** 修复公开绝对姿态转换工具将 180° 目标误转为零旋转的问题。本地独立提交 `d68345b5`，全仓库 lint 与 2047 项 CPU 测试通过；[补丁及验证材料](records/robocasa-half-turn-20261010/verification.json) 已就绪，尚未推送。此修复与下方 RoboTwin 研究结果分开。

> **2026-10-10 合同核查：** 765 条训练演示重算统计及九份部署归一化器均一致，提示、布局和动作缓冲区检查通过；未定位零成功原因。已量化不等间隔记录与训练窗口覆盖，并确认 trace 缺少 planner 状态。详见 [合同核查](EXPANDED_DATA_PLAN_20261006.md#8-训练部署合同核查2026-10-10未启动实验)。暂不启动实验。

> **最新研究状态（2026-10-09）：** 扩充数据研究已完成，540 次尝试 / 531 个有效回合 / 9 个技术失败；S-VAE、raw768、宽头分别 33/177、32/177、24/177，未建立宽头收益。见 [完整结果与审计](EXPANDED_DATA_PLAN_20261006.md#7-完整结果与-cpu-审计2026-10-09)。当前暂不启动实验；下方 10-01 的调度信息为历史记录。
>
> **跨集群调度协议 v1：** [CLUSTER_COORDINATION_PROTOCOL.md](CLUSTER_COORDINATION_PROTOCOL.md)。独立 Git 注册表 + 原子准入；当前全部单元仍由 UIUC Shenlong 的旧队列独占，Torch 未获分配。此协议替代下方历史分配提议。

# NYU Torch deployment — historical `greene/` directory

Current resource instruction (2026-10-19): gpu-01 GPUs1/2 are preferred. Every day
01:00–09:30America/Chicago, any of the16cards on gpu-01/gpu-02 may run prepared
project experiments after30minutes of observed idleness with no compute process.
See [daily GPU admission](nightly_gpu/README.md) for the deployed controller,
idle-history requirements, protected active jobs, and queue registration. Older
resource windows and dated pauses below are historical; scientific study gates
and retained failure records remain in force.

> **2026-10-01 Torch 下一步任务：** [协助 GPU02 评测：先检查资源，再领取未开始的配对任务](GPU02_EVAL_ASSIST_TASKS.md)。GPU02 继续运行；本任务单不转移评测执行权。
>
> **Torch 回报（2026-10-01 23:35 EDT）：** CPU 检查完成，见 [readiness](records/gpu02-eval-assist-20261001-readiness.json)。环境可复用；L40S 预计可立即开始，H200 约 10-02 13:40。**尚未收到私有交接包和新检查点**，单卡预检未运行，未领取任何单元。
>
> **历史分配提议（已替代）：** [GPU02_EVAL_ALLOCATION.md](GPU02_EVAL_ALLOCATION.md)。当前只按跨集群协议 v1 和独立注册表交接；没有提前保留给 Torch 的种子。

> **Current scheduling handoff:** [GPU02 temporal ownership — do not duplicate on Torch/Greene](GPU02_TEMPORAL_HANDOFF.md). Read before submitting temporal jobs. The deployment tables below are historical setup notes.

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
