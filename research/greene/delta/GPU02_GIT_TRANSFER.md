# GPU02 → Greene：用 git 传私有资产（给 GPU02 这边执行）

**背景：** Greene 连不到 GPU02（172.22.224.85 是 UIUC 私网 IP），但两边都能访问 GitHub。
所以在 GPU02 上打一个**逐字节可还原的 delta pack**，推到一个**私有** GitHub repo，
Greene 再 clone 下来还原出原始的 363 个文件（73,274,734,050 bytes）。
原理和校验链见同目录的 [`README.md`](README.md)。

## 不可违反的约束

- **只读源文件。** `pack_delta.py` 只读 `/home/zifanz4/openwam-runtime/{share-eval,assets-source}`，
  不写、不删、不移动。不要停止、暂停、重启 GPU02 worker，不要改 `lane.py`、deadline
  （**2026-09-29 17:48:49 CDT** 不变）或任何 cohort 状态。
- **pack 必须进私有 repo。** 里面有私有权重、scene、config。
  **绝不能**推到 `Exiam6/OpenWAM`（那是 public fork）。`git_push_pack.sh` 检测到
  repo 可以被匿名读取就会中止。
- 用 `nice -n 19 ionice -c3` 跑，免得和正在跑的评测抢 IO。
- 出错不要自动重试或绕过校验，把完整输出带回 Greene。

## 0. 前置条件

```bash
git --version                     # 需要能访问 GitHub
ssh -T git@github.com             # 或 HTTPS + token；需要对私有 repo 有写权限
df -h /home/zifanz4               # --out 所在盘至少留 ~10 GB（预计 pack 4–5 GB）
python3 --version                 # >= 3.6，只用标准库
```

在 GitHub 上新建一个**私有**空 repo，例如 `Exiam6/openwam-greene-transfer`：
Private，不要 README、不要 .gitignore、不要 license。

## 1. 拿到脚本和参考索引（git）

它们在 public fork 的 `research/progress-20260927` 分支上，目录是 `research/greene/delta/`：

```bash
# 已有 OpenWAM clone：
cd <OpenWAM clone> && git fetch origin && git checkout research/progress-20260927 && git pull --ff-only
# 没有 clone：
git clone -b research/progress-20260927 --depth 1 https://github.com/Exiam6/OpenWAM.git ~/OpenWAM-greene
D=~/OpenWAM-greene/research/greene/delta     # 或 <OpenWAM clone>/research/greene/delta
ls $D    # pack_delta.py  git_push_pack.sh  reference-index.json  ...
```

`reference-index.json` 是 Greene 上公开 checkpoint `OpenWAM-Alpha-Sim-RoboTwin-Full/checkpoint_step_118655.safetensors`
的逐 tensor SHA-256 索引，只有哈希，没有权重。它的 reference sha256 是 `d07ff6f8cfb627ebf47313e65af38773fabcb3861d52fdc7c3bc5e7bee33f6fe`。

## 2. 打包

```bash
nice -n 19 ionice -c3 python3 $D/pack_delta.py \
    --root /home/zifanz4/openwam-runtime \
    --reference $D/reference-index.json \
    --out /home/zifanz4/owam-delta \
    --part-mib 95
```

- `--part-mib 95` 保证每个文件都小于 GitHub 的 100 MB 单文件上限，所以不需要 LFS。
- 会读一遍 73 GB，每个源文件都按 `SHA256SUMS` 校验，不一致就中止。
- **先看第一个 checkpoint 那行输出：**
  - `reused-from-reference` ≈ **11.36 GB**：正常，最后 pack 约 4–5 GB。
  - 接近 **0**：私有 text encoder 和公开版不一致。**先停（Ctrl-C）**，回 Greene 换参考源，
    不要推一个 73 GB 的 pack。
- 最后一行 `DONE source … -> pack … GB` 就是实际传输量，记下来。

## 3. 推到私有 repo

```bash
bash $D/git_push_pack.sh /home/zifanz4/owam-delta git@github.com:Exiam6/openwam-greene-transfer.git
```

- 先推 metadata（manifest、`PACK_SHA256SUMS`、share-eval、小文件），然后每批约 1.5 GB 分批推 part。
- 中断了直接重跑，已经提交的会跳过。
- 结束时会打印 `HEAD <sha>`。

## 4. 带回 Greene 的信息（文字就行，不用传文件）

1. repo 地址和最后的 `HEAD` sha
2. `pack_delta.py` 的最后一行（source → pack 大小、reused、dedup）
3. 有错误的话贴完整输出

Greene 这边会执行：

```bash
sbatch --export=ALL,PACK_REPO=git@github.com:Exiam6/openwam-greene-transfer.git \
       /scratch/zz4330/OpenWAM/research/greene/sbatch/delta_unpack.sbatch
```

它会 clone，校验 `PACK_SHA256SUMS`，逐字节还原所有文件，每个都必须等于 GPU02 的原始 sha256，
再跑一遍独立的 `sha256sum -c`，恢复 scene alias，写 `records/asset-transfer.json`。

## 5. 清理（Greene 确认 ALL FILES VERIFIED 之后）

- 删掉私有 repo `openwam-greene-transfer`，因为 GitHub 上不需要长期留私有权重。
- GPU02 上可以删 `/home/zifanz4/owam-delta`，它只是副本，原始文件不受影响。

## 如果 GPU02 也上不了 GitHub

退回手动中转：把 `/home/zifanz4/owam-delta/` 整个目录用任何方式带到 Greene 的
`/scratch/zz4330/openwam-runtime/delta/pack/`，然后不带 `PACK_REPO` 跑 `delta_unpack.sbatch`。
