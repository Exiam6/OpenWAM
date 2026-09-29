# Wan48 原版匹配基线资产 → Greene（在 cm001 / 能读 `/data02` 的机器上执行）

**目的：** 用 endtoend 研究里已经训完的 `wan` 路线（原生 Wan VAE latent，也就是原版 OpenWAM 表征；
295M 参数、105 条轨迹、6000 次更新、3 个 seed）作为**原版匹配基线**。它会在 Greene 上以时间压缩研究相同的
2000 次更新续训，然后评测。这里只负责传资产。

**要传的内容：**

| 内容 | 原始大小 | 校验 |
|---|---|---|
| 3 × `checkpoint_step_12000.safetensors`（wan-seed42/43/44） | 3 × 13.36 GB | **冻结哈希**：`endtoend-20260923/evaluation-ready.json` |
| `endtoend-20260923/normalization_stats.npy`、`data-manifest.json` | 小 | 冻结哈希：`data-preparation.json` |
| 3 个 wan run 目录下的其余文件（config 等；中间 checkpoint 自动跳过） | 小 | 打包时计算 |
| `results/layers-20260922/native-preflight/train-data` | 1.5 GB | 打包时计算 |

预计 pack 约 **3–4 GB**：每个 checkpoint 里冻结的 text encoder / VAE 直接复用 Greene 上的公开 checkpoint，
只传训练过的部分，约 0.6–0.9 GB/个；训练数据 1.5 GB 基本压不动。

## 约束

- **只读** `/data02/zifanz4/openwam-experiments`，不改、不删、不移动。不碰 GPU02 worker 及其评测和 deadline。
- 用 `nice -n 19 ionice -c3` 跑。
- 私有资产优先推到**私有** repo（`git_push_pack.sh` 检测到公开 repo 会拒绝）。如果 owner 这次仍然明确授权用公开临时分支，
  照上次的做法在分支里放 `PUBLIC-TRANSFER.md` 写明授权；Greene 验证完就删分支。
- 出错不要绕过校验，把完整输出带回来。

## 1. 拿脚本

```bash
cd <OpenWAM clone> && git fetch origin && git checkout research/progress-20260927 && git pull --ff-only
D=$PWD/research/greene/delta
ls $D/wan48      # files.txt  SHA256SUMS  WAN48_TRANSFER.md
```

## 2. 先 dry run，确认文件列表和大小

```bash
SRC=/data02/zifanz4/openwam-experiments
ARGS=(--src $SRC --manifest-dir $D/wan48 --reference $D/reference-index.json
      --add-tree endtoend-20260923/training/wan-seed42
      --add-tree endtoend-20260923/training/wan-seed43
      --add-tree endtoend-20260923/training/wan-seed44
      --add-tree results/layers-20260922/native-preflight/train-data)
python3 $D/pack_delta.py "${ARGS[@]}" --out /data02/zifanz4/owam-delta-wan48 --dry-run | tee /tmp/wan48-dryrun.txt | head -40
tail -5 /tmp/wan48-dryrun.txt
```

第一行会给出总文件数和总大小，并列出被跳过的中间 `*.safetensors`。
如果 run 目录里有 **非 safetensors 的大文件**（比如几 GB 的 optimizer 状态 `.pt`），**先停下来**，把 dry-run 输出带回 Greene，
不要直接打包。

## 3. 打包

```bash
nice -n 19 ionice -c3 python3 $D/pack_delta.py "${ARGS[@]}" \
    --out /data02/zifanz4/owam-delta-wan48 --part-mib 95
```

- 3 个 checkpoint 会按**冻结哈希**校验，不一致就中止。
- 每个 checkpoint 那行的 `reused-from-reference` 应该有 **≈ 12 GB 以上**。如果接近 0，先停，回 Greene 换参考源。
- 最后一行 `DONE source … -> pack … GB` 就是实际传输量。

## 4. 推送

```bash
bash $D/git_push_pack.sh /data02/zifanz4/owam-delta-wan48 git@github.com:Exiam6/<私有repo>.git
```

## 5. 带回 Greene 的信息

repo、分支、HEAD sha、pack 大小（`DONE` 那一行）。Greene 执行：

```bash
sbatch --export=ALL,TRANSFER=wan48,PACK_REPO=<repo>,PACK_BRANCH=<branch>,PACK_SHA=<sha> \
       /scratch/zz4330/OpenWAM/research/greene/sbatch/delta_unpack.sbatch
```

它会逐字节还原到 `/scratch/zz4330/openwam-runtime/assets-wan48/`，并检查：
1. 冻结哈希必须和 pack 清单一致，防止传错 checkpoint；
2. 磁盘上每个文件重新计算哈希；
3. 写 `research/greene/records/wan48-transfer.json`。
