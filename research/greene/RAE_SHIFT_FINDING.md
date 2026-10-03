# RAE 的维度相关噪声偏移：我们一直用错了值

日期：2026-10-03。状态：**机制已验证，效果待实测**。下面严格区分这两者。

## 结论先行

RAE 论文最大的单项贡献是一个**维度相关的时间步偏移**，而 OpenWAM **已经实现了完全相同的闭式**，只是把它的值当常量 5.0 用。我们 768 维表征路线的处方值是 **16.43**，一直用的是 **5.0**——欠偏移 3.3 倍，正落在论文所说"噪声腐蚀不足、训练受损"的区间。

这可能解释了为什么我们换到 768 维表征空间后没有看到收益。

## 一、已验证的事实

### 1.1 公式逐字相同

RAE（[Zheng et al. 2025](https://arxiv.org/abs/2510.11690)，ICLR 2026）：

```
t_m = α·t_n / (1 + (α−1)·t_n)        α = √(m/n),  n = 4096
```

OpenWAM `openwam/deploy/denoise_schedule.py::_alpha_shift`：

```python
f_alpha(u) = shift*u / (1 + (shift - 1)*u)
```

以及 `wan/shared/diffusion/flow_match.py:45` 实际施加到 sigma 网格：

```python
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
```

### 1.2 `shift_video` 控制的是**训练**噪声调度，不只是推理

`openwam/model/architectures/base.py::init_training_schedulers` 的 docstring：

> Single source of truth for each stream's α-shift ... so the **discrete training sigma buffer** and the inference denoising trajectory are sampled from the same shifted schedule

训练侧 `compute_loss`（base.py:1089-1097）：

```python
video_timestep_ids = torch.randint(min_tb, max_tb, (B,))
video_sigmas = vb.scheduler.sigmas[video_timestep_ids]
inputs["latents"] = (1 - sigma_bc) * inputs["input_latents"] + sigma_bc * video_noise
```

链条闭合：配置 → `set_timesteps(training=True, shift=...)` → sigma 缓冲 → 每个训练样本的噪声水平。

### 1.3 我们的处方值与实际值

潜变量实测形状（`cache.pt`）：`raw` = (588, **768**, 3, 24, 20)，`svae` = (588, **48**, 3, 24, 20)。

m = 有效数据维度 = T×H×W×C（与是否 patch 化无关，总维度不变）：

| 路线 | m | α = √(m/4096) | 实际配置 | 偏差 |
| --- | --- | --- | --- | --- |
| `svae48_*` | 3·24·20·48 = 69,120 | **4.11** | 5.0 | 接近，合理 |
| `raw768_*` | 3·24·20·768 = 1,105,920 | **16.43** | **5.0** | **欠偏移 3.3×** |

官方 `OpenWAM-Alpha-Sim-RoboTwin-Full` 用 `wan22_vae`、in_dim 48，`shift_video` 同样是 5.0——对它的低维潜变量而言基本正确。**只有高维路线被错配。**

### 1.4 论文给出的效应量

在论文自己的图像生成设定下（Table 4），仅这一项偏移：gFID **23.075 → 4.81**，接近 5 倍。

### 1.4b 论文正文对 m / n / 时机 / 解码器依赖的确认

已从论文 §4.2 正文核实，四点全部吻合我们的套用方式：

| 问题 | 论文原文 | 对我们的意义 |
| --- | --- | --- |
| m 的定义 | "the number of tokens times their dimensionality"；例：256×256 图 + DINOv2-B，16×16=256 token × 768 通道 = 196,608 | 我们 1440 token × 768 = 1,105,920，**算法一致** |
| n = 4096 | "following Esser et al. 2024 in using n=4096 as the base dimension" | 与 SD3 同源，非任意取值 |
| 施加时机 | **training only** | 与我们追踪到的代码路径（训练 sigma 缓冲）完全吻合 |
| 是否依赖解码器 | **独立于解码器**——"would benefit any diffusion model operating in high-dimensional latent spaces, regardless of decoder architecture" | **我们 `pixel_decode=False`、没有 RAE 的训练解码器，不构成障碍** |

最后一条解除了本研究最大的一个疑虑：此前担心缺少 RAE 的像素解码器会让偏移的收益失效，论文明确否定了这种依赖。

### 1.5 上游文档把它当常量

- `assets/openwam_usage_docs/architecture-extension.md:92`：`self._shift_video = 5.0  # scheduler shift, when supported`
- `assets/openwam_usage_docs/train-and-deploy.md:87`：`shift_video: 5.0  # flow-matching scheduler shift`
- 代码 docstring：`shift is None` 时"falls back to each scheduler's template default (Wan/action = 5.0)"

**没有任何一处说明它应当随潜变量维度缩放。** 而 `architecture-extension.md` 恰恰是教人更换编码器的文档——照着它插入高维编码器的人会保留 5.0，静默落进欠偏移区间。

旁证：同一个 `flow_match.py` 里的 `_calculate_shift_qwen_image` **实现了随序列长度插值的 shift**（base_seq_len 256 → max_seq_len 8192）。这个代码库已经知道 shift 该随数据规模变化，只是 Wan 路径没有。

## 二、尚未验证的部分

**我们还没有跑过 α=16.43。** 以上全部是机制推导加代码核实，不是实测结果。可能的落空方式：

- 机器人策略的联合视频-动作流匹配与论文的图像生成设定不同，偏移的收益未必同样显著；
- 我们的规模（4000 步 / 12 层 / 528 窗口）可能小到任何调度改动都被淹没；
- 论文的 m 按 token 网格计数，我们的 token 网格取的是 DINOv3 的 3×24×20；若实际进入 DiT 的 token 划分与此不同，α 会随之变化（但量级不变，仍远大于 5.0）。

§1.4b 已排除的疑虑：m 的定义、n 的来源、"仅训练时施加"、以及**不依赖训练解码器**，四项均经论文正文确认。

## 三、待跑的实验

`rae-shift-20261003`，变体 `raw768_shift16`，脚手架已搭好并验证（模板 diff 只有 `shift_video: 5.0 → 16.4317` 加路径重定向）。

与冻结的 `raw768_native` **严格配对**：同初始化、同 batch 调度、同噪声流、同数据、同种子（42/43/44）、同 4000 步，`shift_action` 保持 5.0 不动（动作流维度在各臂间不变，改它会引入第二个变量）。

预估：1 卡 / 约 60 分钟 / 峰值约 18 GiB。

### 3.1 偏移对训练噪声分布的实际影响

`set_timesteps_wan` 的缓冲即训练实际采样的 σ 分布（`compute_loss` 对 timestep id 均匀采样）：

| shift | mean σ | median σ | σ>0.5 | **σ>0.9** | σ<0.3 |
| --- | --- | --- | --- | --- | --- |
| 4.11（svae48 处方） | 0.7214 | 0.8046 | 80.5% | 31.4% | 9.4% |
| **5.00（实际使用）** | 0.7476 | 0.8336 | 83.4% | **35.8%** | 7.8% |
| 10.00 | 0.8273 | 0.9093 | 91.0% | 52.7% | 4.1% |
| **16.43（raw768 处方）** | 0.8722 | 0.9427 | 94.3% | **64.7%** | 2.5% |

高噪声样本占比从 35.8% 翻到 64.7%。这是"腐蚀不足"的具体量化：768 维模型在 5.0 下近三分之二的训练步看到相对干净的潜变量。

### 3.2 读数必须避开的陷阱

**不能直接比两个臂的 dev video loss。**

`dev_metrics` → `paired_loss` → `model.compute_loss`，而 `compute_loss` 从**模型自己的** sigma 缓冲采样（base.py:1089-1092）。`paired_loss` 里的 `torch.manual_seed` 只固定了抽到的 **timestep id**，而同一个 id 在两个缓冲里对应**不同的 σ**——shift-16 的模型会在系统性更高的噪声上被评测。

直接比较会把"测试更难"读成"模型更差"，产出一个无法解释的结果。

**正确读数**：在**同一组固定 σ** 上评测两个 checkpoint，即绕开模型自带缓冲、显式指定噪声水平，输出 loss–σ 曲线。这比单一标量更有信息量，也能直接看出偏移是否真的改善了高噪声区间的去噪能力——那正是它应当起作用的地方。

闭环成功率仍不作主读数：126 个 cell 仅 13 次成功、两个任务恒零，处于死区。

## 四、无论实验结果如何都成立的一条贡献

即使 α=16.43 在我们的设定下没有带来收益，**上游文档缺口本身是实的**：`shift_video` 必须随潜变量有效维度按 √(m/4096) 缩放，而文档把它呈现为常量 5.0。这条对任何按 `architecture-extension.md` 更换编码器的人都有直接价值，且与我们的策略结果无关。
