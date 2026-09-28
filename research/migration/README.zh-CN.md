# OpenWAM 实验迁移快照

研究目标、最新保存进度与后续执行顺序见 [研究入口](../README.zh-CN.md)。

此分支基于上游 `7c5861e45cfe1339a0323f0e0b03a3316c37971c`。
根目录包含 temporal-recovery-v2 实际部署的 OpenWAM 源码；
`research/experiments/` 包含本地实验仓库的代码、方案、审计、报告和未提交进度。
`snapshot.json` 记录复制时间、原实验提交和逐文件 SHA-256。
历史绝对路径保留，用于审计；它们不代表新机器已经配置好。

## 2026-09-28 迁移完成更新

六个检查点及精确清单363文件已在GPU02完成SHA-256校验（73,274,734,050bytes）。用户授权独立新评测及半小时巡检；当前280/1080完成、3技术失败、1在运行。详见[新study](../experiments/studies/temporal-gpu02-20260928/README.md)。以下原迁移清单/快照保留供追溯，不代表仍需重新传输。

## 历史进度（2026-09-27）

- 时间压缩六组训练已完成。当前比较为 mean/learned × seeds 42/43/44。
- 最近保存的闭环进度见 `../experiments/studies/temporal-recovery-v2-20260927/progress.json`，注意其中时间戳；快照不是实时监控。
- 三表征训练完成，旧评测部分完成并有技术失败；保留所有结果。
- 控制监督 S-VAE 方案在 `../experiments/studies/control-svae-20260926/PLAN.md`，尚未正式训练。
- 旧80次评测及关闭的 layers 窗口不得重跑。迁移不延长现有 deadline。

## 新机器需要另外复制的文件

详见 `external-artifacts.json`。六个最终策略检查点合计 73,138,641,984 字节
（约68.1 GiB），未放进 Git。还需模型资源、tokenizer、时间压缩权重、
S-VAE/PCA、归一化统计、数据、场景和 RoboTwin 资产。
检查点必须连同其父目录的配置文件复制，不能只复制 safetensors。
本目录的 `checkpoint-configs/` 保存配置参考副本。

可在新机器用 rsync 从具备这些共享挂载的源机器拉取（替换占位符）：

```bash
rsync -a --info=progress2 USER@SOURCE:/data02/zifanz4/openwam-experiments/temporal-20260926/training/ /YOUR/DATA/temporal-20260926/training/
```

其他资产按清单复制；先检查符号链接目标是否也已迁移。不要复制 GPU 锁、
PID、编译缓存或直接搬 Python 虚拟环境。当前结果仍在写入，须在明确接管边界后
再做最终结果同步，不能让两台机器执行相同的评测单元。

## 环境与启动前步骤

1. 从你的 fork 克隆并切换 `research/progress-20260927`。
2. 分别创建 policy 与 benchmark 环境。版本参考本目录两个 requirements-lock
   及 benchmark-constraints；RoboTwin/cuRobo 提交和补丁参考 benchmark-setup.json。
   该 setup 文件是历史安装记录，其中旧的验证标记不是当前迁移验证结果。
   CUDA扩展须按目标GPU重新构建；尚未验证这些锁在目标机器可直接安装。
3. 新建独立迁移部署目录；将脚本中的原 research/data/output 路径、解释器、
   主机名、GPU UUID/索引和端口映射到新环境。配置中的模型路径也必须检查。
   修改副本，保留原冻结文件和哈希。禁止对历史方案做全局字符串替换。
4. 对复制的模型与原训练审计核对 SHA-256；记录新路径到旧哈希的映射。
   保持检查点、种子、场景、条件和统计方法不变，另存迁移代码清单。
5. 检查 GPU 空闲/ECC、渲染设备、剩余磁盘（运行守卫至少32GiB空闲）、pause、
   固定 deadline；然后完成原33提示缓存检查、动作一致性和开发场景部署检查。
6. 先列明哪些未执行单元交给新机器，确认原机器不会执行同一单元，再启动。
   现有 `lane.py` 是从头执行的有限队列，**不是跨机器断点续跑工具**。
   不要直接运行它来恢复部分完成的评测，也不要自动重试技术失败。

该分支完成代码/进度归档；目标机器适配、资产传输和部署检查尚未执行。
当前机器上的实验不受此次归档影响。

## 快照验证

```bash
python3 research/migration/verify_snapshot.py
```

通过只说明迁移快照与复制时文件一致，不表示闭环实验已完成或新环境可运行。
