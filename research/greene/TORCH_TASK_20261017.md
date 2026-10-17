# Torch 任务：专家末端轨迹回放（ex4d，约 20 分钟，1 × L40S）

任务 ID：`ex4d-expert-replay-20261017`。调度方：Shenlong。执行方：NYU Torch。
协议：[跨集群调度协议 v1](CLUSTER_COORDINATION_PROTOCOL.md)。计划：[zero-task/PLAN_ex4_20261017.md](zero-task/PLAN_ex4_20261017.md)。

## 0. 为什么

Shenlong（Blackwell）上，把专家演示**自己的**末端轨迹按策略动作通路（`eef20d_to_ee16d` → `take_action(..., 'ee')`）
逐步回放：adjust_bottle 2/2、**handover_block 0/5、place_object_basket 0/1**，规划器 0 次失败。原因：抓取静止段手停在
记录位姿约 11 mm 外（cuRobo 关节终点被控制器精确执行，cuRobo 自身 FK 自洽 → cuRobo 模型 vs 仿真机器人的固定偏差），
手指合拢时把木块推倒。逐步补偿该偏差后 handover 4/5。

但官方 checkpoint 在 Torch 上 handover 20/20、在 Shenlong 上 1/20。如果这个偏差与机器无关，Torch 上应该一样存在。
**唯一要回答的问题：同一回放在 Torch 上是否也失败、偏差是否也存在？**

## 1. 范围

`research/greene/zero-task/demos/` 里 9 条演示（只含 `proprio`、物理步时间戳、`status.json`，已在分支上，188 KB）：
2 × adjust_bottle、5 × handover_block、2 × place_object_basket。不加载任何策略，不需要 checkpoint。

脚本：`research/greene/zero-task/replay_expert_ee.py`（辅助文件 `migration_render.py`、`execution_trace.py`、`collect_scene.py` 同目录）。
在 Torch 上 `configure_robot_paths` 会检测到没有 Shenlong 的配置副本而保持 RoboTwin 原生 cuRobo 配置。

两次运行，环境变量与 job 18771822 / oidn-outcome 的模拟器侧相同（benchmark env、`ROBOTWIN_PATH`、`ROBOTWIN_ENABLE_PLANNER_FALLBACK=1`、
`EXPECTED_RENDER_PCI`、PYTHONPATH 不设），工作目录为 OpenWAM 仓库根：

```bash
Z=$REPO/research/greene/zero-task
export ROBOTWIN_BENCH_DIR=$REPO/benchmarks/robotwin REPLAY_HELPERS=$Z
# A: 原样回放
python -u $Z/replay_expert_ee.py $OUT/replay-torch $Z/demos/*
# B: 每步记录本体末端位姿（只用于量偏差）
REPLAY_PROPRIO=1 python -u $Z/replay_expert_ee.py $OUT/replay-torch-proprio $Z/demos/handover_block-seed4101100
```

预估：1 × L40S，A 约 10 分钟、B 约 2 分钟，显存 < 8 GiB。作业上限 1 h。

## 2. 预注册读法（写于 2026-10-17 17:30 CDT）

| Torch 结果 | 含义 |
| --- | --- |
| handover 0/5 左右，B 中静止段偏差约 11 mm | 偏差与机器无关；官方在 Torch 成功另有原因（其训练数据/标签约定），Shenlong 上的失败不是由这个偏差单独造成 |
| handover ≥ 4/5，B 中偏差 ≤ 3 mm | **偏差是 Shenlong 环境特有的**（bench env torch 2.7+cu128 / warp 补丁 / cuRobo 构建），它同时解释我们所有臂的 0 和官方在 Shenlong 上的崩塌 |
| 其他 | 逐条汇报，不下结论 |

## 3. 汇报

`replay-torch/summary.json`、`replay-torch-proprio/handover_block-seed4101100-proprio.npy`、bench env 的 `torch`、`warp`、`curobo` 版本
（`pip show` / `curobo.__file__` 所在仓库的 commit）。按协议 §4 推到 `research/progress-20260927`。
