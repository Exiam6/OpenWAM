# endtoend 全部评测依赖 → Greene（cm001，一次搬完）

**目的：** 在 Greene 上重新评测 endtoend 的 9 个 checkpoint（Wan48 / SVAE48 / PCA48 × seed 42/43/44），不做任何新训练。
Wan48 的 3 个 checkpoint、训练数据和归一化统计**已经在 Greene 上**（`transfer/wan48-20260929`），这次**不用再传**。
这次把剩下**所有**依赖一次带齐，免得跑第二趟。

## 清单

| 内容 | 路径（相对 `/data02/zifanz4/openwam-experiments`） | 为什么需要 | 校验 |
|---|---|---|---|
| SVAE48 / PCA48 × 3 checkpoint | `endtoend-20260923/training/{svae,pca}-seed4{2,3,4}/output/*/checkpoint_step_12000.safetensors` | 被评测的模型 | **冻结哈希**（`evaluation-ready.json`） |
| 6 个 run 目录的其余文件 | `endtoend-20260923/training/{svae,pca}-seed4*` | config、tokenizer、normalization stats、job 记录；中间 checkpoint 跳过 | 打包时计算 |
| Wan VAE 权重 | `assets/wan22-vae-baseline` | Wan48 config 的 `model_path` 指向这里；**上次漏了** | 打包时计算 |
| DINOv3 | `assets/dinov3-study` | SVAE/PCA 编码器的 `model_path` | 打包时计算 |
| S-VAE / PCA 压缩器 | `endtoend-20260923/compressors-multiview` | `svae_path` 以及 PCA reducer | 打包时计算 |
| 配置模板 | `endtoend-20260923/templates-main` | run config 引用 | 打包时计算 |
| endtoend 冻结代码快照 | `endtoend-20260923/OpenWAM` | `serve_policy.py` / `evaluate_policy.py` 用 `E/OpenWAM` 的 deploy 配置和 robotwin 代码 | 打包时计算 |
| 完整 150 场景 cohort | `endtoend-20260923/scenes` | Greene 只有前 20 个/任务 | 打包时计算 |
| 旧的 270 次评测结果 | `endtoend-20260923/evaluation`、`evaluation-slots`、`recovery/evaluation-20260925` | 单独报告用，**不合并**进新结果 | 打包时计算 |
| 预检证据 | `endtoend-20260923/simulation-preflight`、`deploy-preflight` | 对照 Greene 的初始渲染 | 打包时计算 |
| 公共初始化 | `endtoend-20260923/initialization`（2.1 GB） | 这次评测用不到；为以后的训练（比如控制监督 S-VAE）一起带上，避免再跑一趟 | 打包时计算 |

## 约束

同 `wan48/WAN48_TRANSFER.md`：
- **只读** `/data02`。不碰 GPU02 worker 和它的截止时间。
- **不要**在 `/home/zifanz4/OpenWAM` 里 git pull，那是 GPU02 正在跑的评测的代码目录。
- 用 `nice -n 19 ionice -c3` 跑。
- 私有资产优先推到私有 repo；如果用公开临时分支，放 `PUBLIC-TRANSFER.md` 写明授权。

## 1. 更新工具 clone（和上次同一个独立 clone）

```bash
cd /data02/zifanz4/openwam-greene-tools 2>/dev/null && git pull --ff-only \
  || git clone -b research/progress-20260927 --depth 1 https://github.com/Exiam6/OpenWAM.git /data02/zifanz4/openwam-greene-tools
D=/data02/zifanz4/openwam-greene-tools/research/greene/delta
ls $D/endtoend     # files.txt  SHA256SUMS  ENDTOEND_TRANSFER.md
```

## 2. 组装参数（缺失的路径会列出来，不会中断）

```bash
SRC=/data02/zifanz4/openwam-experiments
ARGS=(--src $SRC --manifest-dir $D/endtoend --reference $D/reference-index.json)
for t in endtoend-20260923/training/{svae,pca}-seed4{2,3,4} \
         assets/wan22-vae-baseline assets/dinov3-study \
         endtoend-20260923/compressors-multiview endtoend-20260923/templates-main \
         endtoend-20260923/OpenWAM endtoend-20260923/scenes \
         endtoend-20260923/evaluation endtoend-20260923/evaluation-slots \
         endtoend-20260923/recovery/evaluation-20260925 \
         endtoend-20260923/simulation-preflight endtoend-20260923/deploy-preflight; do
  if [ -e "$SRC/$t" ]; then ARGS+=(--add-tree "$t"); else echo "MISSING $t"; fi
done
[ -e $SRC/endtoend-20260923/initialization ] && ARGS+=(--add-tree-all endtoend-20260923/initialization) || echo "MISSING initialization"
```

**有 `MISSING` 也继续**，把列表一起带回来。

## 3. dry run

```bash
python3 $D/pack_delta.py "${ARGS[@]}" --out /data02/zifanz4/owam-delta-endtoend --dry-run > /tmp/endtoend-dryrun.txt
grep -v '^ ' /tmp/endtoend-dryrun.txt      # 总量、跳过的中间 checkpoint、最大的 15 个文件、按目录汇总
```

满足下面两条就**直接继续打包**：
- 总量 ≤ 约 **120 GB**，其中 6 个 checkpoint 约 75 GB，大部分会复用 Greene 上已有的公开 checkpoint，不用真的传；
- 除了 checkpoint，没有单个文件 > 3 GB。

否则把 `grep -v '^ '` 的输出带回来，先不要打包。

## 4. 打包并推送

```bash
nice -n 19 ionice -c3 python3 $D/pack_delta.py "${ARGS[@]}" \
    --out /data02/zifanz4/owam-delta-endtoend --part-mib 95
bash $D/git_push_pack.sh /data02/zifanz4/owam-delta-endtoend git@github.com:Exiam6/<私有repo>.git
```

- 6 个 checkpoint 会按冻结哈希校验，不一致就中止。
- 每个 checkpoint 的 `reused-from-reference` 应该有 ≈ 11 GB 以上。

## 5. 带回 Greene 的信息

repo、分支、HEAD sha、`DONE` 那一行，以及第 2 步里所有的 `MISSING`。Greene 执行：

```bash
sbatch --export=ALL,TRANSFER=endtoend,PACK_REPO=<repo>,PACK_BRANCH=<branch>,PACK_SHA=<sha> \
       /scratch/zz4330/OpenWAM/research/greene/sbatch/delta_unpack.sbatch
```

它会逐字节还原到 `/scratch/zz4330/openwam-runtime/assets-endtoend/`，校验冻结哈希，并写 `records/endtoend-transfer.json`。
