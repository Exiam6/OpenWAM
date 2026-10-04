# Torch 任务：提供参照侧测量，隔离跨集群差距的根因

任务 ID：`cross-cluster-rootcause-20261003`。调度方：Shenlong。执行方：NYU Torch。
协议：[跨集群调度协议 v1](CLUSTER_COORDINATION_PROTOCOL.md)。本文件是**任务说明**，不是执行权转移 ACK。

## 0. 为什么是这件事

昨夜测出来：官方发布的 `OpenWAM-Alpha-Sim-RoboTwin-Full`，**同一 checkpoint（sha256 逐字节一致）、同样 60 个场景、所有 serving 参数逐项吻合**——

| | Torch（L40S） | Shenlong（Blackwell） |
| --- | --- | --- |
| 合计 | **58/60 = 96.7%** | **10/59 = 16.9%** |
| `place_object_basket` | 18/20 | **0/19** |

详见 [CROSS_CLUSTER_REPRODUCIBILITY.md](CROSS_CLUSTER_REPRODUCIBILITY.md)。

**在这个差距解释清楚之前，Shenlong 上的任何闭环数字都不可信**——包括我们自己研究的 5/42、5/42、3/42。所有架构比较（含 RAE）都因此阻塞。

**Torch 的角色变了：你是那台能正常工作的机器。** 现在需要你提供参照侧的量，让我们知道该在 Shenlong 上复现什么。

## 1. 第一优先：环境清单（不占卡，几分钟）

最便宜、也最可能直接解释问题的一步。请报告：

| 项 | 怎么取 |
| --- | --- |
| NVIDIA 驱动版本 | `nvidia-smi --query-gpu=driver_version --format=csv` |
| GPU 型号 / compute capability | L40S / sm_89（确认） |
| CUDA runtime / toolkit | `nvcc --version`、`python -c "import torch;print(torch.version.cuda)"` |
| torch 版本 | `python -c "import torch;print(torch.__version__)"` |
| SAPIEN 版本 | `python -c "import sapien;print(sapien.__version__)"` |
| OIDN 是否在 `LD_LIBRARY_PATH`、版本 | 查环境变量与 `sapien/oidn_library` 目录 |
| 渲染后端 | SAPIEN 用的是光栅还是光追？`SAPIEN_*` 相关环境变量有哪些 |
| **`ROBOTWIN_ENABLE_PLANNER_FALLBACK`** | **job 18771822 当时是否设置？** |

最后一项是我唯一没能排除的工具链差异：Shenlong 侧设了 `=1`。理论上它只影响专家规划器、与 `eval_mode` 下的策略回放无关，但未经验证。

Shenlong 侧对照：驱动 **580.95.05**（R580 分支，组内明令不升级）、Blackwell **sm_120**、`TORCH_CUDA_ARCH_LIST=12.0`、`LD_LIBRARY_PATH` 指向 `benchmark-env/.../sapien/oidn_library`、`downloads/` 下有 `oidn-2.3.3`。

## 2. 第二优先：用同一套脚本做渲染测量（约 50 分钟）

你 readiness v1 里报的 MAE 0.0604–0.337 来自 job 18771822 的附带记录。我们在 Shenlong 上是用独立脚本做的全 60 场景普查，为了**苹果对苹果**，请用同一套脚本在 Torch 上跑一遍。

脚本已在本分支/运行时，三步：

1. **重渲染普查**（约 35 分钟，60 场景，不跑策略不计分，attempt-once 不适用）
   `render_survey.py <输出目录>` —— 对每个场景 `setup_demo` + `get_obs`，与参考 PNG 比 MAE，并存下实际渲染。
   环境：benchmark-env，**unset PYTHONPATH**，设 `ROBOTWIN_PATH` / `ROBOTWIN_RUNTIME_ROOT` / `WAM_ROOT` / `LD_LIBRARY_PATH`(oidn) / `EXPECTED_RENDER_PCI`。

2. **潜变量扰动**（约 20 分钟）
   `render_sens_all.py <study根> raw768_native <普查目录> <输出json>` —— 参考与实际渲染分别过视觉编码器，与同任务场景对距离对照。
   环境：policy-env，**设 PYTHONPATH**，**不要覆盖 PATH**。

3. **每步动作偏差**（约 25 分钟，需起策略服务）
   `action_sensitivity.py <port> <输出json>` —— 两种渲染喂生产服务路径，proprio / 指令 / `_study_action_seed` 全相同（`step=0` 使服务端每次预测前 reset 并播种）。

**Shenlong 侧的值，供对照：**

| | Shenlong |
| --- | --- |
| 图像 MAE（60 场景） | min 0.5116 / 中位 0.5718 / max 1.0315 |
| 潜变量扰动 | 占范数 **29.97%**，超过最相似同任务场景对 **60/60** |
| 每步动作偏差 | 中位 **1.31%**，0/60 完全相同 |

如果 Torch 的潜变量和动作偏差明显更小，就把"渲染分歧 → 累积 → 成功率崩塌"这条链补完整。如果 Torch 的也不小，那根因就不在渲染，而在别处（数值/驱动）。**两种结果都有用。**

## 3. 不要做的事

- **不要重跑官方基线。** job 18771822 的 58/60 已经是我们的参照，重跑浪费卡。
- **不要调任何闸门或阈值。** 特别是渲染闸门 1.0 保持不变——`seed-1102008` 在 Torch 上是 0.3032 通过、在 Shenlong 上 1.0315 被拒，这个对照本身是数据。
- **不要改 serving 配置**（compile off / dit_cache off / denoise 10 / horizon 8 / prompt cache 32）。

## 4. 一个更大的选项，待用户裁决

如果根因隔离后确认 Shenlong 的评测环境不可用，那么**最有价值的事是让 Torch 跑我们自己三个变体的闭环评测**——那样架构比较（含 RAE 的全部问题）就能在一台能工作的机器上重做。

**2026-10-03 更新：所有者已授权并建立私有仓库 `Exiam6/openwam-greene-transfer`。** 传输按 `delta/GPU02_GIT_TRANSFER.md` 的 delta 机制走，清单见 `delta/rae-policy/`，源 116 GB 经参考索引去重后预计只传训练权重。Shenlong 侧打包就绪、待执行；推送后会在本分支记录 `HEAD` sha。Torch 侧届时按 §4 用 `delta_unpack.sbatch` 还原。

## 5. 汇报方式

按协议 §4 走 `research/progress-20260927`，与 readiness v1 同样的形式。环境清单（§1）可以先单独推，不必等渲染测量跑完——它最便宜也最可能直接给出答案。
