# Torch 任务：官方 checkpoint 闭环基线

任务 ID：`official-baseline-20261002`。调度方：Shenlong。执行方：NYU Torch。
协议：[跨集群调度协议 v1](CLUSTER_COORDINATION_PROTOCOL.md)。本文件是**任务说明**，不是执行权转移 ACK。

## 0. 为什么这次不需要资产传输

上一轮 `gpu02-eval-assist-20261001` 卡死在资产交付（Torch 的 readiness v1 记录 `delivery_channel: none received`）。**这次不存在那个阻塞**：

| 需要的东西 | 来源 | 是否需要 Shenlong 交付 |
| --- | --- | --- |
| 官方 checkpoint（24.8 GB） | **HuggingFace 公开发布** | **否**，Torch 自己下 |
| RoboTwin 2.0 任务资产 | **公开** | **否**，Torch 自己下 |
| 15 个场景定义 | 我们的录制，**合计 1 MiB** | 待定，见 §3 |
| 评测脚本 | 本仓库 | 否，随分支获得 |

## 1. 目标

用官方发布的 `OpenWAM-Alpha-Sim-RoboTwin-Full` checkpoint，在**我们 `rae-policy-20261001` 研究使用的同一批场景种子**上跑干净闭环，得到逐场景成功/失败。

**要回答的问题**：`handover_block` 和 `place_object_basket` 在一个真正能干活的策略上是否可做。

我们三个臂在这两个任务上**全部为零**（0/15 与 0/12，跨 3 变体 × 3 种子）。两种可能：

- 官方也接近零 → 任务/评测环境的问题，我们的规模不是主因，后续规划需重做；
- 官方跑得动 → 确认是规模问题（我们 4,000 步 / 12 层，官方 118,655 步 / 30 层）。

## 2. Checkpoint（已在 Shenlong 侧核验）

```
python scripts/download_assets/download_openwam_checkpoints.py \
    --family alpha --name OpenWAM-Alpha-Sim-RoboTwin-Full --yes
```

落地 `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full/`，自包含（config.yaml + 权重 + tokenizer + normalization_stats.npy），无需认证。

**请在 Torch 侧核对这两项后再跑：**

| 项 | 期望值 |
| --- | --- |
| 文件 | `checkpoint_step_118655.safetensors` |
| 字节数 | `24813767464` |
| SHA256 | `d07ff6f8cfb627ebf47313e65af38773fabcb3861d52fdc7c3bc5e7bee33f6fe` |

Shenlong 已下载并逐字节核验通过，SHA256 与历史 manifest 记录一致。

官方架构（供交叉检查）：编码器 `wan22_vae`，DiT `in_dim/out_dim 48`、`dim 3072`、`num_layers 30`、`num_heads 24`、`ffn_dim 14336`。

## 3. 场景（需要你确认交付方式）

必须跑的 15 个种子，与我们的研究严格配对：

| 任务 | 场景种子 |
| --- | --- |
| `adjust_bottle` | 1100000, 1100001, 1100002, 1100003, 1100004 |
| `handover_block` | 1101000, 1101001, 1101002, 1101003, 1101004 |
| `place_object_basket` | 1102002, 1102006, 1102007, 1102008, 1102009 |

**不要**改用 `baseline-official/protocol.json` 里提的 20 个新场景——那样只是"协议相似"，不是逐场景配对。

每个场景目录含 `collection-config.json`、`initial-replay.json`、`initial-{head,left,right}_camera.png`。

**场景已随本分支交付，无需另行传输。** 位置 `research/greene/scenes/`，布局与运行时一致：

```
research/greene/scenes/<task>/manifest.json
research/greene/scenes/<task>/seed-<N>/{collection-config.json,initial-replay.json,initial-*.png}
```

把评测脚本里的 `E` 指向 `research/greene/` 即可（运行时原值是 `assets-source/temporal-20260926`）。

交付的是**全部 60 个场景**（每任务 20 个）而非仅上表 15 个，合计 7.6 MiB、303 个文件，原仓库里的符号链接已全部解引用为真实文件。多给的 45 个场景有两个用处：

1. 上表 15 个用于与我们 126 个 cell 的**逐场景严格配对**；
2. 每任务 20 个正好对应 `baseline-official/protocol.json` 原本提的规模，能把"这两个任务到底可不可做"的统计功效提高 4 倍——鉴于我们三个臂在这两个任务上**全部为零**，这个功效是需要的。

**两批请分开汇报**，不要混为一个成功率。

## 4. 评测配置

沿用 `baseline-official/protocol.json`：compile off、DiT cache off、`denoise_steps 10`、`inference_horizon 8`、prompt cache 32。
动作 RNG：`SeedSequence([scene, 42, 811])`，与 seed-42 各路配对。
条件只跑 `clean`。步数上限用任务原生值：`adjust_bottle` 400、`place_object_basket` 700、`handover_block` 800。

## 5. 需要回报的内容

每个场景一条记录：场景种子、任务、成功与否、步数、耗时、`initial_render_mae`、`initial_state` 摘要、动作种子、GPU 型号与 UUID、Slurm job id。

**务必记录 `initial_render_mae`**——见下一节。

技术失败单列，不计为策略失败，不自动重试。

## 6. 一个必须一并测量的东西

Shenlong 侧发现：场景物理状态复现到 1e-6（proprio 和全部物体位姿断言均通过），**但同一状态的渲染在全部 135 个 cell 上系统性偏离参考 0.52–1.03 个 uint8 级**，跨 9 次独立运行离散度为 0.00，差异局部化，带符号均值≈0（不是曝光/gamma 偏移）。

当前闸门是 `MAE > 1.0` 即判技术失败，`place_object_basket` 的 `seed-1102008` 恰好越线（1.0315），在全部 9 个臂上被一致丢弃。

Torch 是**第三个渲染环境**。它的 `initial_render_mae` 分布是一个独立数据点：

- 若 Torch 也落在同一区间 → 是采集/评测配置差异（`eval_mode`/`is_test`/`render_freq`/`save_data`），与机器无关；
- 若 Torch 明显不同 → 说明还有机器相关分量。

**若 `seed-1102008` 在 Torch 也越过 1.0 而被拒，请如实记录，不要放宽闸门。**

## 7. 执行前置

按协议 §2–§3：先做 CPU 侧依赖与资产校验；确认实际拿到 GPU（Slurm job/allocation id、型号、UUID、可用时长）；`sbatch --test-only` 的预测不算就绪。单卡预检只用训练场景，不碰计分场景。

本任务不涉及训练种子的所有权转移——它是一个**新的、独立的**评测任务，不从 `gpu02-eval-assist-20261001` 的任何 unit 中切分，因此不需要那条注册表的 transfer。

## 8. Shenlong 侧状态

GPU5 当前被他人占用（`xzeng28`），我们这边无法立即跑官方基线——这是指派给 Torch 的直接原因。
阶段 2 的 `vae48_native` 对照臂留在 Shenlong，待 GPU5 空出后执行。
