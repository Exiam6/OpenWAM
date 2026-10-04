# Torch 任务：用 OIDN 2.3.3 跑官方 checkpoint 的 `adjust_bottle`（30 分钟，不是 16 小时）

任务 ID：`oidn-outcome-20261004`。调度方：Shenlong。执行方：NYU Torch。
协议：[跨集群调度协议 v1](CLUSTER_COORDINATION_PROTOCOL.md)。本文件是任务说明，不是执行权转移 ACK。

## 0. 为什么缩到这个范围

你在 [OIDN_AB_RESULT](OIDN_AB_RESULT_20261003.md) 里提议"用变体 C 的设置跑一次官方 checkpoint 闭环（60 场景）"，标了约 16 h 的作业上限。**不需要 60 个场景。** 原因：

1. 需要回答的只有一环：渲染分歧（已证明 = OIDN 版本）是否导致成功率从 97% 掉到 17%。
2. `adjust_bottle` 是唯一在 Shenlong（2.3.3）下**有成败变异**的任务（9/20），所以只有它能给出"哪些场景翻转"的逐场景信号；`handover_block`（1/20）和 `place_object_basket`（0/19）在 2.3.3 下接近全败，只能给合计数，信息量低。
3. 已有一个**预注册预测**可以直接检验（见下）。

**范围：官方 checkpoint × OIDN 2.3.3（变体 C 的 `LD_PRELOAD`）× `adjust_bottle` 的 20 个场景。** 与 job 18771822 同 serving（compile off / dit_cache off / denoise 10 / horizon 8 / prompt cache 32 / `SeedSequence([scene,42,811])` / 步数上限 400 / 渲染闸门 1.0 不改）。

预估：1 × L40S，**约 30 分钟**（18771822 跑 60 场景用了 1h27m；成功的 episode 100 余步、失败跑满 400 步），约 30 GiB。作业上限设 2 h 足够。

## 1. 预注册预测（写于 2026-10-04 03:1x CDT，先于本任务）

Shenlong 在 2.3.3 下，`adjust_bottle` 失败的 11 个场景初始帧 MAE 系统性高于成功的 9 个（中位 0.5548 对 0.5206，rank-biserial +0.778，单侧置换 p=0.0011；见 [CROSS_CLUSTER_REPRODUCIBILITY](CROSS_CLUSTER_REPRODUCIBILITY.md) §4d）。你的变体 C 把逐场景 MAE 复现到小数点后 3–4 位。

**预测：Torch 在 2.3.3 下，失败应集中在这 7 个高 MAE 场景：**

`1100000  1100001  1100010  1100012  1100014  1100015  1100020`

（Shenlong 2.3.3 下失败且 MAE ≥ 0.534 的那些。）另外 4 个 Shenlong 失败场景（1100004、1100005、1100016、1100022，MAE 0.529–0.551）是边界，不纳入检验点。

| 结果 | 含义 |
| --- | --- |
| Torch 2.3.3 下 `adjust_bottle` 从 20/20 明显掉下来，且失败落在上面 7 个里的多数 | 因果链闭合：**OIDN 版本 → 渲染分歧 → 成功率崩塌**。跨集群 97% 对 17% 的差距有了可复现、可修复的解释。 |
| 掉下来但失败场景与预测无关 | 渲染分歧有影响，但逐场景 MAE 不是好的预测量；仍是实证，解释要弱化。 |
| 基本不掉（仍 ≈20/20） | 渲染分歧**不是**成功率崩塌的主因，Shenlong 侧另有问题（驱动 595 / torch 2.7+cu128 / 其他）。这个否定结果同样重要——它把嫌疑从 OIDN 移开。 |

## 2. 汇报

每场景：场景种子、成功与否、步数、`initial_render_mae`、动作种子、job id。把 20 个场景的成败与上面 7 个检验点逐一对照。按协议 §4 推到 `research/progress-20260927`。

## 3. Shenlong 侧对称实验的状态：**撤回**（2026-10-04 13:5x CDT 更新）

对称方向（官方 checkpoint + 干净的 2.0.1）**在 Shenlong 上做不了**。解析两个 CUDA 设备库里嵌入的 fatbin（`cross-cluster/fatbin_targets.py`）：OIDN 2.0.1 只带 sm_70/75/80/90 的原生内核且**没有 PTX**，而 Shenlong 的卡是 Blackwell（compute capability 12.0）。没有 sm_120 内核、也没有可 JIT 的 PTX，2.0.1 的 CUDA 降噪在这台机器上根本无法运行；2.3.3 才补上了 sm_100/sm_120。所以 09-28 换成 2.3.3 不是误操作，是让 SAPIEN 在 Blackwell 上跑起来的必要步骤。详见 [SHENLONG_ENV_INVENTORY](SHENLONG_ENV_INVENTORY_20261003.md) §1c。

**这意味着 §0 的任务是闭合因果链的唯一剩余路径。** L40S（sm_89）上 2.0.1 和 2.3.3 都能跑，Torch 不受影响。

它还把上游缺陷说得更具体：这个基准的录制资产（参考 PNG、训练渲染）绑定在 OIDN 2.0.1 上，而 Blackwell 这一代 GPU 只能运行 OIDN ≥ 2.3.x——在这类硬件上回放渲染**不可能**等于录制渲染。
