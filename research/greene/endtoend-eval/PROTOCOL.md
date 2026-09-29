# Greene 重评 endtoend 匹配表征对照 — 协议草案（用户已确认，未冻结）

创建于 2026-09-29。**用户于 2026-09-29 确认本草案，不做修改**（不加高斯 0.04 条件，场景数不变）。 资产到齐、开发门槛通过、用户确认之后，再冻结成 `protocol.json`，
附代码哈希和绝对截止时间。冻结之前不跑任何评测场景。

## 问题

在**完全匹配**的 295M 三任务设置下（同初始化、同 105 条训练轨迹、每个 seed 的 batch 流 SHA-256 相同、
同为 6000 次更新），压缩表征（SVAE48、PCA48）的闭环成功率是否不同于**原版 OpenWAM 表征 Wan48**？
这是 endtoend 研究原本要回答的问题。它的闭环评测因为基础设施故障停在 270/5400，这里在 Greene 上**独立重评**。

## 被评测的模型（不训练）

| 路线 | seed | checkpoint（固定为最终的 12000 microsteps / 6000 次更新） | 冻结 sha256 |
|---|---|---|---|
| wan | 42 / 43 / 44 | `endtoend-20260923/training/wan-seed*/output/*/checkpoint_step_12000.safetensors` | `b9cf3c71…` / `5514b15f…` / `e9dd53f4…` |
| svae | 42 / 43 / 44 | 同上，路线换成 svae | `4cbe7625…` / `1424a145…` / `a05abc58…` |
| pca | 42 / 43 / 44 | 同上，路线换成 pca | `7504a93a…` / `2502bfd7…` / `cef39a8b…` |

完整哈希见 `research/greene/delta/{wan48,endtoend}/SHA256SUMS`，来源是 `evaluation-ready.json`。
每个 checkpoint 加载前都会重新校验。

## 评测

- **场景：** sealed endtoend cohort 中每个任务的**前 20 个 accepted 场景**。和时间压缩研究、Greene 官方基线用的是**同一批 60 个场景**，所以三项结果能放进同一张表。
- **任务：** adjust_bottle、handover_block、place_object_basket。
- **条件：** `clean`、`gaussian_sigma_0.10`、`head_camera_yaw_5deg`，和时间压缩研究一致。噪声和相机位姿的生成规则与原协议相同，所有方法、所有 seed 都用同一套。
- **回合数：** 9 个 checkpoint × 3 任务 × 20 场景 × 3 条件 = **1620**。
- **部署：** compile 关、DiT cache 关、denoise_steps 10、inference_horizon 8，使用原生任务步数上限，每回合墙钟上限 600 s。服务和评测代码来自 `research/experiments/scripts/endtoend/`，配合 `endtoend-20260923/OpenWAM` 代码快照，只做路径适配，改动另存为 Greene 版本并记录哈希。
- **动作随机数：** 和时间压缩研究一样，`SeedSequence([scene, training_seed, 811])`。
- **渲染门槛：** 初始状态误差 ≤ 1e-6；每个相机的初始渲染 MAE ≤ 1（uint8），对照原机器的参考 PNG。不达标记为技术失败，然后继续下一个场景。

## 统计与判定（结果出来之前固定）

- **主要比较：**
  1. **SVAE48 − Wan48**、**PCA48 − Wan48**（压缩表征 vs 原版表征），clean 和两个扰动条件分别报告；
  2. **PCA48 − SVAE48**（原 endtoend 的主要比较），作为次要比较。
- **指标：** 三个任务等权的成功率差。场景 × seed 交叉配对 bootstrap，10000 次，seed 426；三个 seed 的效应分别报告。
- **声称有收益的条件：** 1620 个有效结果全部齐，技术失败不能被悄悄删掉；95% CI 下界 > 0；三个 seed 的效应同号。
  否则报告为"有限 / 负面 / 不确定"。**任何结论都不能外推到 5B 或官方 50 任务基准。**
- **局限（事先写明）：** 这 60 个场景已经在旧的 270 回合评测和时间压缩评测里用过，不是新的独立测试数据；3 个 seed 的精度也有限。

## 不合并、不重试

- 旧的 270 回合（cm009）原样保留，**单独报告**，不和本次结果合并，也不从两者中挑一个。
- 技术失败原样保留，**不自动重试**；截止时间冻结后不延长。
- 和 GPU02 不重叠：本方案只评测 endtoend 的 checkpoint；GPU02 评测的是时间压缩的 checkpoint，两边没有共同的评测单元。

## 资源（估算）

每回合约 80 s，共约 **36 L40S·h**。同时最多用 6 张 L40S（和组里共享的 48 卡上限不冲突），墙钟约 6–8 小时。
每个 (checkpoint, 任务, 条件) 是一个有限的 Slurm 作业，跑完就退出，由 `update_progress.py` 自动更新进度。

## 冻结前的门槛

1. endtoend 资产到齐：`records/endtoend-transfer.json`，冻结哈希一致。
2. 每条路线在开发场景上跑一次部署门槛：服务能起来、动作有限、提示缓存一致。
3. 在 1 个 L40S 上跑 1 个开发场景的完整回合，确认耗时。
4. ~~用户确认~~（已确认 2026-09-29）→ 写 `protocol.json`，内容包括参数、代码 sha256、截止时间。
