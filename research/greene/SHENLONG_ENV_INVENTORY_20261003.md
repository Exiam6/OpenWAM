# Shenlong 环境清单：OIDN 被换成了 2.3.3

日期：2026-10-03。回应 [TORCH_ENV_INVENTORY_20261003.md](TORCH_ENV_INVENTORY_20261003.md) 的要求。

## 0. 结论

Torch 问"Shenlong 实际加载哪个 OIDN"。答案是 **2.3.3，不是 SAPIEN 自带的 2.0.1**。

```
sapien/oidn_library/libOpenImageDenoise.so.2
  -> openwam-runtime/downloads/oidn-2.3.3.x86_64.linux/lib/libOpenImageDenoise.so.2.3.3
```

符号链接建立于 **2026-09-28 04:33**。从 `/proc/self/maps` 实测确认，在我们的评测环境下进程加载的是：

```
downloads/oidn-2.3.3.../libOpenImageDenoise.so.2.3.3
downloads/oidn-2.3.3.../libOpenImageDenoise_core.so.2.3.3
```

`sapien/oidn_library/` 目录里 2.0.1 与 2.3.3 两套 `.so` 混放，soname 链接指向后者。

## 1. 三机对照

| 项 | Shenlong | Torch |
| --- | --- | --- |
| **OIDN（实际加载）** | **2.3.3**（符号链接被改指 downloads） | **2.0.1**（SAPIEN 自带） |
| `LD_LIBRARY_PATH` | 显式指向 `sapien/oidn_library` | **未设** |
| GPU | RTX PRO 6000 Blackwell，sm_120 | L40S，sm_89 |
| **驱动** | **595.71.05** | 610.43.02 |
| SAPIEN | 3.0.0b1 | 3.0.0b1（相同） |
| bench torch | **2.7.0+cu128 / CUDA 12.8** | **2.4.1+cu124 / CUDA 12.4** |
| planner fallback | `=1` | `=1`（已确认，**此项排除**） |

SAPIEN 版本相同，planner fallback 相同。**差异集中在 OIDN、驱动、torch/CUDA 三处。**

## 2. 为什么 OIDN 是头号嫌疑

降噪器直接决定光追渲染的输出。我们测到的渲染分歧形态是：**局部化、带重尾、带符号均值≈0、跨运行完全确定**（9 次运行离散度 0.00）。这正是"同一场景经不同降噪实现"的签名，而不是曝光/几何差异。

量级上也对得上：Shenlong 图像 MAE 中位 0.5718，Torch 0.1019，约 5.6 倍。

## 3. 决定性检验（约 15 分钟，1 卡）

把符号链接改回 SAPIEN 自带的 2.0.1，重渲染若干场景，看 MAE 是否从约 0.57 降到约 0.10：

```bash
D=.../site-packages/sapien/oidn_library
ln -sfn "$D/libOpenImageDenoise.so.2.0.1" "$D/libOpenImageDenoise.so.2"
# 跑 cross-cluster/render_survey.py，对比 MAE
```

若 MAE 回落 → **根因确认是 OIDN 版本**，然后用官方 checkpoint 重跑 20 个 `adjust_bottle` 场景（约 30 分钟），看成功率能否从 9/20 回到 20/20。
若 MAE 不变 → 根因在驱动或 torch/CUDA，需另外隔离。

**这个改动会影响共享环境里任何正在跑的渲染任务，所以等用户点头再做。**

## 4. 一个独立的警告

Shenlong 驱动现在是 **595.71.05**。组内记录明确要求保持 **580.95.05（R580 分支）**，因为 **595.x/R590 已知会让 Isaac Sim 在这张卡上 `ERROR_DEVICE_LOST` 崩溃**。驱动已被升级到被警告的分支。

这与跨集群差距可能相关也可能无关，但本身就该让负责人知道。

## 5. 脚本已随本分支提供

Torch 报 §2 被阻塞（脚本不在分支上）——是我的遗漏，现已补上 `research/greene/cross-cluster/`：

| 脚本 | 用途 | 环境 |
| --- | --- | --- |
| `render_survey.py` | 60 场景重渲染 + MAE，存实际渲染 | benchmark-env，**unset PYTHONPATH** |
| `render_sens_all.py` | 全场景潜变量扰动 vs 场景间距离 | policy-env，**设 PYTHONPATH**，不覆盖 PATH |
| `render_sens3.py` | 单场景双分辨率版（上者的前身） | 同上 |
| `action_sensitivity.py` | 每步动作偏差（需策略服务） | policy-env，同上 |
| `eval_sigma_curve.py` | 共享 σ 网格上的 loss–σ 曲线（RAE 偏移研究用） | policy-env，同上 |

踩过的坑：策略服务**不要覆盖 PATH**（会静默死掉、日志零字节）；模拟器必须 **unset PYTHONPATH** 并设 `ROBOTWIN_*`；闸门文件要带 `code_sha256`。
