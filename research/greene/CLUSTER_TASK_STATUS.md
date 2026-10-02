# Shenlong GPU5 / Torch 任务状态

核实时间：2026-10-02 02:26 CDT。本文件的每一项都来自本次对 `shenlong-gpu-02` 的实时核查，不是对话记录的转述。

## 1. 一句话结论

Shenlong 的 GPU5 正在跑评测，队列**还剩 2 个 cell**，预计 **03:18 CDT 前后全部跑完**；按 [调度协议 v1](CLUSTER_COORDINATION_PROTOCOL.md) §3 末段，本轮 **Torch 不接任何计分单元**。

## 2. GPU5 实时状态（已核实）

| 项 | 实测值 |
| --- | --- |
| 节点 | `shenlong-gpu-02.cs.illinois.edu`，GPU 5 |
| 利用率 / 显存 | 76%，18690 MiB / 97887 MiB |
| 启动器 | `eval-launch.py evaluate` pid 540327，已运行 4h46m |
| 队列进程 | `evaluate.py` pid 541690 |
| 父服务 | `openwam-rae-policy-eval-after-training-20261001.service`，active |
| 正在跑的 cell | `raw768_native` / seed 44 / `handover_block` / clean |
| 暂停标志 | 4 个 `pause_paths` 全部不存在 |

GPU5 的显存和算力当前只有本任务在用。`shenlong-gpu-01` 上 5 号卡的占用属于其他用户（`hongchix` 的 `skill_benchmark`），与本任务无关——**两台节点的 GPU 编号不要混为一谈**。

## 3. 评测矩阵进度

矩阵 = 3 变体 × 3 种子 = 9 个 cell。训练侧 9/9 全部 `complete=True, strict_reload=True`。

| seed | svae48_native | raw768_native | raw768_wide |
| --- | --- | --- | --- |
| 42 | 完成 22:14:37 | 完成 22:52:43 | 完成 23:30:29 |
| 43 | 完成 00:08:54 | 完成 00:46:52 | 完成 01:25:11 |
| 44 | 完成 02:01:51 | **运行中**（约 02:02 起） | 未开始（队列最后一个） |

7 个 cell 已 `attempts_finished`，三个任务组 exit_code 均为 0。单 cell 实测耗时 36m40s–38m18s，平均约 37.7 分钟。

## 4. 时间线与硬限制

| 时刻（CDT） | 事件 |
| --- | --- |
| ~02:40 | `raw768_native-seed44` 预计结束 |
| ~03:18 | `raw768_wide-seed44` 预计结束，矩阵跑完 |
| ~03:20 | `evaluate.py` 汇总 ledger 后退出，**GPU5 转为空闲** |
| 06:34:09 | 父服务 `RuntimeMaxUSec=11h45min` 到期（起算 10-01 18:49:09） |
| 06:38:16 | `protocol.json` 绝对 deadline |

以更早的 06:34:09 为准。两个限制都远晚于 03:18，当前队列**不会被时限截断**。

## 5. GPU5 跑完之后

冻结协议内已无剩余计分工作：矩阵 9 个 cell 本轮全部会被尝试，`automatic_retries: false`，cell 最多尝试一次。`control-svae-expanded-20261001` 的采集阶段也已 `exit_code: 0` 结束。

因此 03:20 之后到 06:34 之间约 **3 小时 14 分**的空窗，只能靠**开一个新实验**来填。这超出当前冻结协议的范围，需要先定方案再占卡——协议 §3 明确写了"不为填满卡而重训/重跑"。候选方向和预估见第 7 节。

## 6. Torch 已经交了，本轮不再需要更多

Torch 在 **2026-10-02 00:12 EDT** 推了 [readiness v1](records/gpu02-eval-assist-20261001-torch-readiness-v1.json)（提交 `2cbf69e6`，来自 `torch-login-a-1` 登录节点）。这份报告是合规的，先记录它交了什么：

| 项 | Torch 报告值 |
| --- | --- |
| 协调器测试 | 13 tests OK（CPU，登录节点） |
| 观察到的注册表 | `138ebb7`，revision 0，全部 unit owner=`gpu02`，`mode=legacy` |
| GPU allocation | `allocated: false`，`requested: false` |
| 不申请的理由 | 协议 v1：资产未校验前不申请 GPU，不空占卡 |
| 资产校验 | `false` —— **一份都没收到** |
| 交付通道 | `none received`；私有仓库 `Exiam6/openwam-greene-transfer` 不存在，也没有新的 `transfer/*` 分支 |
| 已 claim 单元 | `[]`（空） |
| QOS | `gpu48`，单用户上限 16 卡，当前占用 0 |

两点值得注意：

- **Torch 没有空占卡，这是对的。** 它明确写了 `sbatch --test-only` 只是预测不是就绪，并据此拒绝提前申请——正是协议 §1 和 §2 要求的行为。
- **卡住 Torch 的是 Shenlong 这边。** `missing_inputs` 四项里，checkpoint + SHA256SUMS、固定源码/配置/归一化统计、私有 cell-ID 对照表都该由 Shenlong 交付，逐单元准入 runner 也明确标注 `owned by Shenlong`。Torch 没有拖延。

**但本轮这些都已经不需要补了。** 三个种子都已开始，协议 §3 末段生效：不拆开正在跑的种子，Torch 本轮 claim 0 个计分 cell，Shenlong 自己跑完。所以：

1. Shenlong **不需要**交付资产，也**不需要**推 ack JSON——没有可转移的未尝试种子。
2. Torch 报告里被阻塞的"资产 SHA256 校验"和"单卡预检"两项，本轮**作废**，不用补。
3. Torch 本轮**不需要再提交任何东西**。它的 readiness v1 已经是完整的收尾记录。

Torch 报告里 H200 的 `--test-only` 预估开始时间是 10-02 14:23 EDT，本来就在硬截止之后，这一路本轮不可用；L40S 预估 00:38 EDT 可用，但因为没有资产，申请了也只是空占卡。两条路本轮都不走。

如果下一轮要真正让 Torch 接计分单元，缺的是同一份东西：Shenlong 先建交付通道（私有 transfer 分支 + SHA256SUMS），再部署逐单元准入 runner，然后才谈 transfer。这是下一轮的前置条件，不是本轮的待办。

## 7. 空窗期如果要开新实验

需要你先定方向。按单卡 RTX PRO 6000 Blackwell、03:20 起算到 06:34 硬停，可用窗口 **3h14m**，预算应留 20 分钟余量，即**实际可用约 2h50m**。

注意 Torch 侧窗口更紧：它报告的 `new_work_deadline` 是 07:28 EDT（= 06:28 CDT），比 Shenlong 的 06:34 CDT 还早 6 分钟。

任何方案启动前仍要满足 §3/§4：独立的协议冻结、cell 清单、注册表领取，且不得写入本轮已冻结的 `rae-policy-20261001` 结果目录，避免污染已完成的配对比较。

## 8. 本文件没有覆盖的

- Torch 侧的实时 Slurm 状态未独立核查；第 6 节的 Torch 数据全部转引自它自己推送的 readiness v1，报告时间 00:12 EDT，距本文件核实时间已过约 2.4 小时。
- 未对已完成 cell 的策略结果做任何统计解读；`attempts_finished` 与 `complete` 只表示技术上跑完，不表示策略成功。
