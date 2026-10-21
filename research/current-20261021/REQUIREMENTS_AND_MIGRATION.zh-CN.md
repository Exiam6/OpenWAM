# 8×A100 / 可选H100：资源需求与迁移步骤

用户已提供候选资源：8张A100，可能还有H100。A100显存版本、H100台数、节点/
集群地址、调度方式、独占时长和渲染能力尚未确认。本文件给出可执行的准备流程；
不将尚未验证的8张计算卡直接记为8个可用的完整仿真worker。

## 资源与部署选择

| 项目 | 当前实测 / 目标要求 |
|---|---|
| 目标工作 | 发布checkpoint原生评测；每worker一个policy server加一个RoboTwin client，不需要8卡DDP重训 |
| 源GPU | NVIDIA RTX PRO6000 Blackwell Server Edition，97,887MiB；driver580.95.05 |
| 显存 | 源server约25GiB、simulator约6GiB。属于特定任务实测，非跨卡峰值保证；A10040GB先验证峰值，80GB余量更大 |
| GPU占用 | 独立推理worker；若使用MIG或共享，先验证可见设备、显存与图形能力，记录分区 |
| 渲染 | 必须能创建SAPIEN3原生`rt`相机并产出三路真实RGB，单独运行`nvidia-smi`或CUDA矩阵乘法不算通过 |
| CPU/RAM | 起始规划每worker4–8 CPU、16–32GB RAM，8worker节点约32–64 CPU、128–256GB RAM；须按实测调整 |
| 磁盘 | checkpoint目录约24.84GB；另外完整RoboTwin资产/纹理、两个环境、每worker编译缓存、日志/视频；建议预留200–300GB以上，先实测 |
| 运行磁盘守卫 | 现有admission要求80GiB空余，运行要求60GiB；迁移若不同需单独登记原因 |
| 网络 | 权重/资产下载或从源机同步；server/client若分节点，需可达WebSocket端口、低延迟和稳定连接 |
| 时间 | 计划外推267.68GPUh，280GPUh有界阶段，截止2026-10-28 18:20:27 UTC；8个同速完整worker理想纯算约33.5h，不是A100实测ETA |

**A100/H100的图形兼容性是独立门槛。** SAPIEN区分raster与ray-tracing shader；
本研究原生场景使用`rt`，不能为迁移方便改成raster或CPU/planner fallback。
A100图形支持会受设备/驱动/虚拟化方式影响，不能根据CUDA可用就假设Vulkan可用。
参见 [SAPIEN渲染文档](https://sapien-sim.github.io/docs/user_guide/rendering/customize_shaders.html)
和 [NVIDIA的A100/Vulkan讨论](https://forums.developer.nvidia.com/t/nvidia-a100-support-vulkan/241422)。
后一来源包含不同驱动/部署的报告，不作为所有A100裸机一律不支持的证明。

候选方案：

1. **同一分配内完整worker。** 单卡同时承担policy和原生client，只有真实场景预检通过才扩到8卡。
2. **分开部署。** A100/H100运行policy server；具备已验证Vulkan/RT能力的RTX/L40S类
   GPU运行原生RoboTwin/cuRobo，沿官方WebSocket接口连接。保持科学配置，另外记录
   主机、GPU、网络、随机性和吞吐差异；先验证再登记。不能先假设一张渲染卡能喂满8个server。

CUDA架构需按目标重编译：A100计算能力8.0、H100为9.0；
现有脚本写死`TORCH_CUDA_ARCH_LIST=12.0`和Blackwell CUDA路径，不能照搬。
来源：[NVIDIA compute capability列表](https://developer.nvidia.com/cuda/gpus)。

## 需要固定的科学设置

- 官方OpenWAM commit：`782e3133f674449e2986754cfc5f7c4e204835a2`。
- RoboTwin commit：`0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`。
- 模型：`OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full`，`checkpoint_step_118655.safetensors`。
- checkpoint SHA256：`d07ff6f8cfb627ebf47313e65af38773fabcb3861d52fdc7c3bc5e7bee33f6fe`。
- 整个checkpoint目录一起迁移：模型文件、`config.yaml`、`normalization_stats.npy`、
  `tokenizer/`。目录当前约24,835,229,433bytes，不能只复制safetensors。
- 保持10步sync denoise、DiT cache0.99/max_skips3、compile启用、native full horizon、
  cuRobo、seed0、unseen instructions、原生候选筛选和每任务步数上限。
- clean/randomized各50任务×100accepted回合；所有pilot和中断部分回合排除于正式成绩。

官方推理与评测入口必须来自上述官方commit，**不要把本交接分支根目录当成最新版官方运行代码**。
本分支以用户fork/main为基底，仅同步当前计划和必要研究脚本；完整上游另开checkout并固定。

## 两个Python环境

| 环境 | 当前源机实测版本 | 迁移判断 |
|---|---|---|
| policy | torch2.7.0+cu128，transformers5.17.0，diffusers0.40.0，numpy2.2.6，safetensors0.8.0 | 源机可运行组合；不是承诺目标机器pip一条命令可重建 |
| benchmark | Python3.10，torch2.7.0+cu128，SAPIEN3.0.0b1，warp1.13.0，mplib0.2.1，numpy1.26.4 | 源机Blackwell兼容偏差已记录；官方验证torch为2.4.1 |
| planner | RoboTwin自带cuRobo v0.7.8；120份源码与安装来源逐字节核对 | 对目标SM重编译；不得启用替代planner |

新A100/H100平台优先尝试官方验证的benchmark软件组合；若采用源机组合或其他版本，
记录差异并通过相同场景与动作/观测预检。不要混用policy/benchmark环境，也不要直接
复制虚拟环境、Blackwell编译缓存或假定旧requirements-lock仍是当前环境。

## Git与大文件获取

```bash
git clone --branch handoff/openwam-plan-20261021 \
  git@github.com:Exiam6/OpenWAM.git OpenWAM-plan
cd OpenWAM-plan
# 先阅读此目录，再单独准备官方runtime。
git clone https://github.com/OpenWAM-Official/OpenWAM.git ../OpenWAM-official-runtime
git -C ../OpenWAM-official-runtime checkout 782e3133f674449e2986754cfc5f7c4e204835a2
```

权重来源：[官方HuggingFace发布目录](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full)。
也可从源机已验证目录同步，路径见`status.snapshot.json`。RoboTwin对象资产、embodiment、
description资源、seen/unseen纹理均须齐全；官方背景纹理来源：
[TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)。

同步后核对所有哈希，检查符号链接实际目标；embodiment YAML内旧绝对路径只做
路径迁移，数值配置保持一致。资产和原始证据通过单独传输，不提交Git大文件。

## 迁移与验收顺序

1. **确认资源。** 主机/集群、A10040/80GB、H100型号/台数、是否独占、Slurm/容器、
   运行时长、出网、共享盘和可用Vulkan图形GPU。先确认真实可见的物理/逻辑GPU。
2. **准备隔离部署。** 新目录、两个环境、官方源/模型/资产、路径映射和目标SM；
   保留源runtime，不做全局替换或修改运行中的protocol/hash。
3. **单worker技术预检。** 哈希、CUDA/cuRobo、原生RT三相机、finite20D状态/动作、
   真实server/client、planner和原生成功规则；检查没有在其他物理GPU留下额外渲染占用。
   做固定任务clean/randomized预检及视频审阅，记录跨设备数值差异。
4. **先移交少量未领取任务。** 不直接执行`register.py`或`all`队列。
   现有脚本含源机路径、UUID、device arch、systemd及deadline，需要新适配和登记。
5. **先跑一个正式100回合任务。** 验证吞吐、显存、count/log/_result/video一致性和资源开销。
   新机预检回合不计正式成绩。确认后再按登记分配扩到2、4、8worker。
6. **独立完整审计。** 新增固定selector把移交后的每行准确映射到原/新机器的一份完整结果；
   核对每行100回合、原生Success!/Fail!与计数、视频hash/解码，以及总共100行。
   报告完整分数和任务bootstrap；不能把不同版本、重复或部分回合混入结果。

## 跨机器唯一所有权：先移交，再启动

快照里的`eligible_for_transfer_at_snapshot`只表示拍快照时没有claim，**不是移交许可或锁**。
源调度器仍可能在下一次轮询领取它；运行中的任务不能被新机重复执行。

源端移交流程：短暂停止我们自己的controller服务（独立GPU worker继续运行），
重新读取claims；仅把选定仍未领取的job原子移动到带批次编号的transferred目录，
保存job hash、科学身份、原allocation/deadline和新owner，再恢复controller。
目标端得到这份冻结移交记录后才登记和执行。保留旧job和所有claims，禁止删除或重置。
如果移交期间已被领取，该行撤回移交。迁移不自动延长原deadline或总预算。

推荐按整行(task,mode,100回合)分工，按登记max_seconds做负载均衡；任务种子与候选
序列不分片、不择优。原clean/adjust_bottle已指定ex8-e恢复，应显式移交其恢复身份。
保存目标机原始日志、视频、environment、命令、checkpoint/source/asset hash；
把完整证据带回审计端后，另冻结完整selector，原selector和中断记录保留。

## 接下来仍缺的信息

A100显存版本；节点/SSH或集群名称；H100台数与型号；是否独占及可用时长；
SAPIEN/Vulkan/RT预检结果；可用CPU/RAM/磁盘；server与client间端口/网络条件。
这些尚未提供，不在本次Git推送中假定已完成目标部署。
