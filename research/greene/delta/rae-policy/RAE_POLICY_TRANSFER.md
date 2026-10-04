# rae-policy-20261001 的三个变体 → Torch（私有仓库）

**目的：** 在 Torch 上重新评测我们自己的三个变体（`raw768_native` / `raw768_wide` / `svae48_native` × seed 42/43/44），
不做任何新训练。背景见 [CROSS_CLUSTER_REPRODUCIBILITY.md](../../CROSS_CLUSTER_REPRODUCIBILITY.md)：
官方 checkpoint 在 Torch 上 58/60、在 Shenlong 上 10/59，所以 Shenlong 上的闭环数字都不可信，
架构比较必须搬到一台能工作的机器上重做。

## 通道

私有仓库 `git@github.com:Exiam6/openwam-greene-transfer.git`，2026-10-03 由所有者建立。
已验证：匿名访问 404（私有）、空仓库、SSH 可达。**这是 `GPU02_GIT_TRANSFER.md` 规定的唯一通道**；
`git_push_pack.sh` 检测到仓库可匿名读取会中止。

## 清单（`files.txt` / `SHA256SUMS`，相对 `/home/zifanz4`）

| 内容 | 路径 | 校验来源 |
|---|---|---|
| 9 个 checkpoint | `openwam-runtime/rae-policy-20261001/runs/<变体>-seed<N>/checkpoint_step_4000.safetensors` | 各 run 的 `result.json` 里保存时记录的 `checkpoint_sha256`；打包时逐文件重算核对 |
| 每个 run 的 `result.json` / `initialization.json` / `final-dev.json` | 同上目录 | 打包时计算 |
| `protocol.json` / `eval-protocol.json` / `code-freeze.json` / `normalization_stats.npy` | `openwam-runtime/rae-policy-20261001/` | 打包时计算 |
| 配置模板（三变体） | `openwam-runtime/rae-policy-20261001/templates/` | `--add-tree`，打包时计算 |
| 冻结源码树（含 RAE 宽头） | `openwam-rae-research/` | `--add-tree`，打包时计算。注意这是**磁盘上的冻结工作树**，不是 commit `e983676`——`code-freeze.json` 的哈希对应的是前者 |

`cache.pt`（1.4 GB，训练用的潜变量缓存）**不传**：闭环评测用不到。

## 预期体积

源 **116.26 GB**（dry-run 实测，1002 个文件）。真实传输量取决于冻结张量与公开参考
`checkpoint_step_118655.safetensors`（sha256 `d07ff6f8…`，参考索引已核对一致）的去重：
只有训练出来的 DiT + action 权重（每个 run 300M–417M 参数）和不匹配的冻结张量进包。
**以 `pack_delta.py` 最后一行打印的数字为准。** 若第一个 checkpoint 的 `reused-from-reference` 接近 0，停下，不要推。

## 命令（在 gpu-02 上）

```bash
# 打包：只读源文件，低 IO 优先级，不占卡
cd /home/zifanz4 && setsid nohup nice -n 19 ionice -c3 python3 owam-delta-tools/pack_delta.py \
  --root /home/zifanz4/openwam-runtime --reference owam-delta-tools/reference-index.json \
  --out /home/zifanz4/owam-rae-delta --part-mib 95 \
  --manifest-dir /home/zifanz4/openwam-runtime/share-rae-policy --src /home/zifanz4 \
  --add-tree openwam-runtime/rae-policy-20261001/templates --add-tree openwam-rae-research \
  > owam-rae-delta.log 2>&1 < /dev/null &

# 推送：先 metadata，再每批约 1.5 GB 分批推 part；中断可直接重跑
bash owam-delta-tools/git_push_pack.sh /home/zifanz4/owam-rae-delta git@github.com:Exiam6/openwam-greene-transfer.git
```

## Torch 侧还原

按 `GPU02_GIT_TRANSFER.md` §4：`sbatch --export=ALL,PACK_REPO=git@github.com:Exiam6/openwam-greene-transfer.git delta_unpack.sbatch`。
每个还原文件必须等于本清单里的原始 sha256。**九个 checkpoint 的哈希同时记录在各自的 `result.json` 里**，可以独立交叉核对。

## 评测时的注意

- 三个变体的 `templates/<变体>/config.yaml` 里 `normalization_stats_path` 与 `output_path` 是 Shenlong 的绝对路径，Torch 需改写。
- 服务与评测配置与官方基线相同：compile off / dit_cache off / denoise 10 / horizon 8 / prompt cache 32，动作 RNG `SeedSequence([scene, 42, 811])`。
- 用本分支 `research/greene/scenes/` 下的同一批场景，这样与 Shenlong 的 126 个有效 cell 逐场景配对。
- 渲染闸门 1.0 **不改**；逐场景记录 `initial_render_mae`。

## 状态

- 2026-10-03 20:5x：清单、工具、dry-run 全部就绪；私有仓库已建；**打包尚未执行**（见对话记录）。
