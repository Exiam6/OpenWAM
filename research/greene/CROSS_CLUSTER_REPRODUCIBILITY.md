# 同一 checkpoint，跨集群成功率 96.7% 对 16.9%

日期：2026-10-03。状态：**已实测，60/60 完整**。

## 0. 结论

官方发布的 `OpenWAM-Alpha-Sim-RoboTwin-Full`，**同一个 checkpoint（sha256 逐字节核验一致）、同样的场景、同样的全部 serving 配置**：

| 任务 | **Torch（L40S）** | **Shenlong（Blackwell）** |
| --- | --- | --- |
| `adjust_bottle` | **20/20** | **9/20** |
| `handover_block` | **20/20** | **1/20** |
| `place_object_basket` | **18/20** | **0/19** |
| **合计** | **58/60 = 96.7%** | **10/59 = 16.9%** |

`place_object_basket` 在 Shenlong 上**完全失效**（0/19），而 Torch 是 18/20。第 20 个 cell 是技术失败，见 §3。

这不是我们自己的小模型，是官方发布的、在 Torch 上几乎满分的 checkpoint。

## 1. 受控到什么程度

Torch 的运行记录与我们逐项对照：

| 项 | Torch | Shenlong |
| --- | --- | --- |
| checkpoint sha256 | `d07ff6f8…` | 同（两边各自独立核验） |
| 场景 | 同 60 个录制场景 | 同 |
| condition | clean | clean |
| compile / dit_cache | False / False | False / False |
| denoise_steps / inference_horizon / prompt_cache | 10 / 8 / 32 | 10 / 8 / 32 |
| action_rng | `SeedSequence([scene, 42, 811])` | 同 |
| step_limits | 400 / 800 / 700 | 同（取自原生 `env.step_lim`） |
| render_gate | per-camera MAE ≤ 1.0，未改 | 同 |
| preflight | 已通过 | 已通过 |
| **GPU** | **L40S（sm_89）** | **RTX PRO 6000 Blackwell（sm_120）** |

**唯一记录在案的差异是机器。** 每一个 serving 参数都一致。

## 2. 失败形态：没有中间态

| 任务 | 成功步数 | 失败步数 |
| --- | --- | --- |
| `adjust_bottle` | 103, 104, 107, 111, 111, 111, 112, 117, 128 | **全部恰好 400**（上限） |
| `handover_block` | 352（仅一次） | **全部恰好 800**（上限） |

成功时干脆利落（100 余步），失败时**无一例外跑满上限**——即从未触发成功条件，不是"差一点"。这是"早期被带偏后再也回不来"的形态，与逐步偏差累积一致。

## 3. 技术失败精确复现

唯一的技术失败是 `place_object_basket seed-1102008`，`head_camera` MAE **1.0315234**，被 1.0 闸门拒。

这与我们此前在 135 个 cell 上测到的值**完全相同**，也与 Torch 上的 **0.3032（通过）**形成对照。同一场景、同一闸门、两台机器、两个结果。

## 4. 已测到的机器相关分量

渲染分歧是我们**唯一实测过**的机器相关差异（见 [RENDER_DIVERGENCE_RESULT.md](RENDER_DIVERGENCE_RESULT.md)）：

| | Shenlong | Torch |
| --- | --- | --- |
| 图像 MAE（60 场景） | 0.5116 – 1.0315（中位 0.5718） | 0.0604 – 0.337（中位 0.1019） |
| 潜变量扰动 | 占范数 29.97% | 未测 |
| 每步动作偏差 | 中位 1.31%（0/60 完全相同） | 未测 |

物理状态在两边都复现到 1e-6（proprio 与全部物体位姿断言均通过），**分歧只在渲染**。

## 5. 必须守住的因果边界

**已确立**：同一 checkpoint、同样场景、同样全部 serving 配置，跨机器成功率 97% 对 23%。**这本身就是一个严重的基准可复现性问题，与根因是什么无关。**

**未确立**：是渲染分歧导致的。本实验**隔离不出根因**。候选包括：

- 渲染路径差异（唯一经实测的，每步 1.3% 动作偏差 × 400–800 步）；
- GPU 架构（sm_120 对 sm_89）在策略前向里的数值差异；
- 驱动（Shenlong 为 580.95.05）、CUDA/cuDNN、SAPIEN/OIDN 构建差异。

渲染分歧是**有实测支撑的头号嫌疑**，但"头号嫌疑"不等于"已证明"。

一个未排除的工程差异：Torch 用 `baseline-official/scripts/evaluate_policy.py`，我们用 `rae-policy-20261001/evaluate_policy.py`。两者 diff 过，逻辑一致、只差路径与闸门文件；但我们的 `simrun` 设了 `ROBOTWIN_ENABLE_PLANNER_FALLBACK=1`，Torch 是否设置未知。该开关理论上只影响专家规划器、与 `eval_mode` 下的策略回放无关，但未经验证。

## 6. 这对我们之前的结论意味着什么

`rae-policy-20261001` 的全部闭环数字（三臂 5/42、5/42、3/42，两个任务恒零）都是在 **Shenlong 这台机器上**测的。既然官方 checkpoint 在同一台机器上从 97% 掉到 23%，**我们自己那些接近零的成功率，有多少是模型不行、有多少是机器环境，现在无法分离。**

这不推翻"我们的规模不足"这个判断（4,000 步对 118,655 步的差距是独立成立的），但它**削弱了所有基于 Shenlong 闭环数字的定量比较**。

## 7. 下一步

1. ~~跑完 `place_object_basket`~~ —— 已完成，0/19，整组失效。
2. **隔离根因**：最直接的做法是在 Torch 上也测渲染 MAE 与每步动作偏差（我们已有脚本），再看能否用一台机器上的两种渲染设置复现成功率差异。
3. **若确认**：这是一个值得上报 OpenWAM 的评测完整性问题——任何跨机器复现该基准的人都会遇到，而且因为没人记录 `initial_render_mae`，它是静默的。
