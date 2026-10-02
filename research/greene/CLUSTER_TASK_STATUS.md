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

## 6. Torch 本轮应该提交什么

协议 §3 末段：*"如果 Torch 就绪时所有种子都已开始，则不拆开正在跑的种子。"*

三个种子（42/43/44）**都已开始**，seed 44 的最后一个变体也会在约 03:18 前结束。所以本轮触发的是该条款：**Torch 不 claim、不启动任何计分 cell**，Shenlong 把本轮跑完。

Torch 应当提交的是以下四项，都不需要计分所有权：

1. **资源实况记录。** 实际拿到的 Slurm job/allocation ID、GPU 型号与 UUID、可用时长、当前 QOS 与剩余配额。`sbatch --test-only` 的预测、PENDING 作业、历史 6 卡吞吐**都不算**资源就绪（§1）。拿不到卡就如实写拿不到。
2. **资产校验结果。** 对已收到的资产逐文件 SHA256 比对 `code-freeze.json` 与 `source-sha256.json`，提交比对结果（含不一致项）。
3. **单卡预检证明。** 只用指定训练场景，不跑任何计分场景、不训练，有时限。L40S 优先（sm_89，与冻结的 `build_arch 8.9` 一致）；H200 仅在更早可用且另行验证兼容性时选用。提交预检日志与实测吞吐。
4. **无需所有权的分析。** 对已有结果做完整性核验（ledger 的 planned/attempted/valid/technical_failures 对账），不重跑、不重训。

提交路径按 §4：走 `coordination/gpu02-eval-assist-20261001` 分支的 `registry.json`，经 `coordination/coordinator.py` 做 fast-forward push；**禁止 force push、手动 merge/rebase、绕过工具改 owner**。文档和聊天记录不是锁。

Shenlong 这边本轮**不需要**推 ack JSON 转移执行权——没有可转移的未尝试种子了。

## 7. 空窗期如果要开新实验

需要你先定方向。按单卡 RTX PRO 6000 Blackwell、03:20 起算到 06:34 硬停，可用窗口 **3h14m**，预算应留 20 分钟余量，即**实际可用约 2h50m**。

任何方案启动前仍要满足 §3/§4：独立的协议冻结、cell 清单、注册表领取，且不得写入本轮已冻结的 `rae-policy-20261001` 结果目录，避免污染已完成的配对比较。

## 8. 本文件没有覆盖的

- 未核查 Torch 侧的实时 Slurm 状态（本次只核查了 Shenlong 节点）。
- 未对已完成 7 个 cell 的策略结果做任何统计解读；`attempts_finished` 与 `complete` 只表示技术上跑完，不表示策略成功。
