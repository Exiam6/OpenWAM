# GPU02 / Torch 跨集群调度协议 v1

任务：`gpu02-eval-assist-20261001`。本协议替代旧 allocation 文档中的提议；**不是执行权转移 ACK**。
目标：同一计分单元最多尝试一次；已拿到的算力持续推进；完整配对比较保持在同一硬件类别上。

## 1. 当前生效边界

- GPU02 GPU 5 的旧串行启动器仍运行，全部三个种子仍归 GPU02。它尚未接入下面的逐单元准入检查，因此注册表标记为 `mode=legacy`；工具拒绝任何 claim/transfer。
- Torch 尚未验证本轮资产，也没有可据以转移任务的实际 GPU allocation。`sbatch --test-only` 的预测、PENDING 作业和历史 6 卡吞吐都不算资源就绪。
- 原截止时间保持：2026-10-02 **06:38:16 CDT / 07:38:16 EDT**。现有 systemd 服务更早在约 **06:34:09 CDT** 到期，以更早限制为准。新调度准入在硬截止前 600 秒停止，不得沿用旧提议中 10 月 3 日的等待期限。
- 调度协议、工具、测试和所有权哈希可在本 fork 的调度分支协同。研究实现、资产、具体模型结果及 ID 对照表不随这次调度更新公开；本协议不扩大已有发布授权。

## 2. 唯一事实来源与任务粒度

| 层级 | 单位 | 规则 |
| --- | --- | --- |
| 跨集群所有权 unit | 一个完整训练种子：全部对照组、任务、条件和场景 | 一次只属于一个集群；开始任何单元后不再跨集群迁移 |
| 集群内 worker shard | 一个种子下一个任务：全部对照组、条件和场景 | 一个 worker 独占；同一种子固定 GPU 型号，可在该型号多张卡间并行 |
| 最小 cell | 协议指纹 + 检查点 SHA256 + 训练种子 + 任务 + 条件 + 场景 | 计分最多尝试一次；技术失败和中断也不能自动重跑 |

所有权注册表独立保存在 fork 的 **`coordination/gpu02-eval-assist-20261001` 分支 `registry.json`**。
进度文档仍在 `research/progress-20260927`。文档、聊天、排队记录不是锁；没有注册表确认不能启动。
`legacy` 阶段 cell 状态仅为带时间戳的历史快照，**不能据此判定可交接**；owner 始终有效。
公开 cell ID 是 canonical JSON（排序键、无额外空白）的 SHA256。协议指纹合并训练协议、评测协议和源码冻结清单的文件哈希。
unit manifest 是排序后的全部 cell ID 列表的同样 SHA256。私有 ID 对照表应随获授权的资产提供，不能凭旧矩阵猜测。

## 3. 无重叠交接顺序

1. **Torch 准备，GPU02 继续。** 先做 CPU 依赖和资产检查；资产 SHA256 齐备才申请有时限的单卡预检。预检只用指定训练场景，不跑正式计分场景。L40S 优先；H200 仅在更早可用且兼容性验证通过时选用。同一任务不要同时在两种分区抢跑。
2. **就绪后才谈任务。** Torch 回报实际 Slurm job/allocation ID、GPU 型号/UUID、可用时长、当前 QOS/用户剩余配额、各 unit manifest 校验和预检证明。排队不锁种子。等待资产时取消不必要的 GPU 申请，避免空占卡。
3. **GPU02 在完整任务组边界排空旧启动器。** 先禁用下一组的启动入口，让当前组自然结束；确认所有子进程结束、旧队列/重启器不会再次启动，并保存停止证明、完整 cell 清单和最终结果哈希。不能只改正在运行的 Python 文件，或只杀一个 simulator 留着父启动器继续派活。
4. **安装并验证准入适配器后 adopt。** 新 runner 必须执行本协议的逐 cell 检查，使用原冻结实验源码；所有路径/启动改动单独记录。`adopt` 要求完整、停止状态的 inventory。任何尝试过的单元均不得变回 pending。适配器目前待部署，不能通过填写一个布尔值替代实际验证。
5. **原 owner 发起 transfer。** 仅从当时仍全部未尝试的队尾种子中挑选；GPU02 先有生效的排除规则和停机证明，Torch 资源仍新鲜/就绪，然后原子转移 owner 并增加 epoch。候选种子不是预约；GPU02 可以在 Torch 就绪前正常推进它们。
6. **Torch 收到远端确认再 claim。** claim 成功取得随机 token；每个 cell 再执行 start-cell 并确认 push 成功，之后才启动进程。旧 owner/token 不得继续领取。资源失效、网络失败或提交失败均不启动。

如果 Torch 就绪时所有种子都已开始，则不拆开正在跑的种子。Torch 转去做无需计分所有权的环境准备、结果完整性核验或已有数据分析；不为填满卡而重训/重跑。本轮 GPU02 继续完成。

## 4. 原子提交、状态与故障

[`coordination/coordinator.py`](coordination/coordinator.py) 是 Python 标准库实现的状态转换工具，不包含私有评测代码，也不直接启动 GPU 进程。
每次读取远端 HEAD → 校验/生成状态 → 普通 fast-forward push。两个调用基于同一 HEAD 竞争时，最多一个提交被接受。
**状态分支禁止 force push、手动 merge/rebase 和绕过工具改 owner。** 冲突方重新 fetch，重新判断是否还能申请；不能直接重放旧提交。
同一 worker 的调用串行化，独占自己的专用 checkout；初始化 checkout 也必须串行。多个 worker 各用一个目录。
这是遵循协议的协作进程之间的准入控制，不是防御有写权限者恶意修改 Git 的安全边界。

| 状态 | 允许的下一步 |
| --- | --- |
| `legacy` unit | 旧 owner 继续；实际排空、适配器通过验证后才 adopt |
| `pending` shard | owner + 实际就绪 allocation 可 claim |
| `claimed` / `running` shard | 当前 token 的 worker 心跳、启动尚未尝试的 cell、提交结果 |
| `complete` / `failed` / `interrupted` cell | 终态，不允许自动重试；complete 仅表示技术上完成，不表示策略成功 |
| `done` shard | 不再领取 |

- worker 每 **60 秒** heartbeat；资源报告每 **60 秒**刷新。心跳超过 **180 秒**或资源报告超过 **300 秒**停止启动新 cell。已有 cell 在原 600 秒上限和原作业时限内结束；Git 不可达时结果先可靠写本地。
- **超时不释放所有权。** 必须确认旧 worker/所有子进程停止、待运行 Slurm 作业已取消、旧准入失效，再执行 recover。处于 running 的 cell 记 interrupted，其余未尝试 cell 才可继续。失联且无法确认停止时保持所有权，报告阻塞。
- 如果 push 成功但回执丢失，先核对 registry 的 operation/event/attempt；不能猜测失败后重启。已记 start-cell 的单元保守视为尝试过。
- cell 启动拿到的 attempt ID 随结果保存；finish-cell 校验 token/attempt，并只提交结果文件 SHA256。硬件、代码版本、耗时、原生成功判定等详细记录保留在授权结果存储。
- 公共状态约每 10 分钟汇总，故障/完成/资源就绪/需交接立即更新；无需给用户重复发送无变化通知。此处是 runner/watcher 的运行要求，**本次没有另外创建常驻 watcher 或启动自动任务**。

## 5. 算力分配与吞吐策略

| 资源 | 初始策略 | 扩展条件 |
| --- | --- | --- |
| GPU02 GPU 5 | 保持当前队列；新 runner 先 1 worker | 同卡双 worker 先短预检：总吞吐提升至少 15%、峰值显存不超过 80%、无 OOM、协议和行为校验通过；最多 2 worker |
| Torch L40S/H200 | 实际分到卡且资产就绪后，先 1 卡 canary | 并发 = min(实际分配、核实的用户可用配额、任务许可上限、就绪 shard 数)；本轮保守上限 2 张 GPU，不按历史 6 并发申请 |
| 排队/无资产 | CPU 准备、检查清单和结果分析 | 不占计分所有权，不保留无工作可做的 GPU allocation |

每个 unit 内固定硬件类别；worker 各自使用独立端口、日志/临时目录、模拟器实例和进程组，遵守 Slurm 的 `CUDA_VISIBLE_DEVICES`，不硬编码物理卡号。
双 worker 使用的是同一张物理 GPU，不是新增 GPU 许可。还要核对 CPU/RAM、渲染和存储吞吐；不能只看显存剩余来加并发。

分配不预设“两边各一半”：用**本轮预检/已完成任务**的单位耗时、模型加载成本、实际剩余窗口估算各 cluster 完工时间。
每次优先把最长的可交接完整种子交给能最早完成它的已就绪资源；只有预计交接/加载后仍能缩短完工时间才转移。
集群内先跑耗时较长的 task shard，完成即领取下一个。旧实验的 80 秒/回合不能直接替代本轮估时。
剩余窗口容不下一个受限回合、或 Slurm 即将到期时停止领取；不降低去噪步数、不减场景、不挑选性重试来追求占用率。

## 6. 调用与对接

工具位置来自进度分支；注册表 checkout 必须另外创建：

```bash
python3 research/greene/coordination/coordinator.py \
  --remote git@github.com:Exiam6/OpenWAM.git \
  --checkout "$HOME/openwam-coord-worker-1"
```

上述不带 request 时只读状态。写请求用 `--request /absolute/local/request.json`；请求文件不用提交到公共分支。
例如实际准备妥当的资源声明（示例值必须换成真实证明）：

```json
{
  "action": "advertise", "actor": "torch", "operation_id": "unique-report-id",
  "evidence": {
    "allocation_id": "slurm-ACTUAL-JOB-ID", "hardware": "L40S",
    "slots": 1, "workers_per_gpu": 1, "verified_gpu_limit": 2,
    "allocated": true, "qos_verified": true, "assets_verified": true,
    "smoke_passed": true, "runner_admission_verified": true,
    "verified_manifests": ["EXACT-UNIT-MANIFEST-SHA256"],
    "receipt_sha256": "LOCAL-VERIFICATION-RECEIPT-SHA256"
  }
}
```

`receipt_sha256` 指向本地保存的真实验证记录；工具检查声明、哈希和状态，不会自己 SSH 验证远端 Slurm/进程。这是适配器和操作方必须完成的前置工作。
资源声明不是领取。在 adopt/transfer 后，owner 才能发送：

```json
{
  "action": "claim", "actor": "torch", "unit": "seed-44", "shard": "task-01",
  "worker": "slurm-JOB-ID-worker-0", "operation_id": "unique-claim-id",
  "evidence": {"allocation_id": "slurm-ACTUAL-JOB-ID"}
}
```

claim 的 JSON 回执包含 token。后续 heartbeat、start-cell、finish-cell、finish-shard 请求必须携带相同 actor/unit/shard/worker/token。
start-cell 另带 cell ID；finish-cell 另带 attempt ID、`evidence.outcome`（complete/failed）和 `evidence.result_sha256`。
runner 只有在 start-cell **退出码为 0 且远端确认**后启动一次实际进程。状态分支冲突时重新读取、校验；不因“之前 claim 成功过”跳过逐 cell 检查。

```bash
python3 -m unittest discover -s research/greene/coordination/tests -v
```

测试覆盖真实 Git 竞争、终态不可重跑、旧 token 失效、心跳/资源过期、配额限制、硬件一致性、旧队列隔离及有证明的交接/恢复。

## 7. 两边下一步

**GPU02：** 保持现有评测；准备独立的逐 cell 适配器，在非计分场景验证；拿到 Torch 就绪回报后，在组边界执行排空/清点/adopt，届时才决定有没有完整未开始的种子可转移。没有完成这些条件前，不推送执行权 ACK。

**Torch：** 拉取本协议和工具；更新真实资产缺项、QOS、队列和可用窗口；运行 CPU 协调器测试；资产齐备后才申请单卡预检。请勿把当前 seed-43/44 候选视为分配，也不要自行将 legacy 改为 managed。ready 时提供验证证明，再由 GPU02 发起 transfer。
