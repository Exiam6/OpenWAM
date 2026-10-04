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

## 1b. 进一步核实：Shenlong 上根本没有完整的 2.0.1

`sapien/oidn_library/` 里 **所有** `*.so.2.0.1` 文件都是指向 `downloads/oidn-2.3.3` 的符号链接，不只是 soname 链接：

```
libOpenImageDenoise.so.2.0.1       -> downloads/oidn-2.3.3.x86_64.linux/lib/libOpenImageDenoise.so.2.3.3
libOpenImageDenoise_core.so.2.0.1  -> downloads/oidn-2.3.3.x86_64.linux/lib/libOpenImageDenoise_core.so.2.3.3
```

两者 sha256 相同（`9ac9dcd1…`），`DT_NEEDED` 相同（都依赖 `core.so.2.3.3`）。

wheel 安装时的 `sapien-3.0.0b1.dist-info/RECORD` 记录 `libOpenImageDenoise.so.2.0.1` 原始 sha256 为 `9h88Er7HAe88…`（131,481 字节）；
现在磁盘上该路径算出来是 `msnc0YMYyuh5…`。**原始 2.0.1 被整体替换掉了**，不是只改了一个链接。

**已恢复一份干净的 2.0.1**：从 PyPI 重新下载原始 wheel `sapien-3.0.0b1-cp310-cp310-manylinux2014_x86_64.whl`（49.6 MB，只解压不安装），
取出三个 OIDN 库放到 `openwam-runtime/oidn-2.0.1-pristine/`。核验：`libOpenImageDenoise.so.2.0.1` sha256 = `9h88Er7HAe88…`，
**与 RECORD 完全一致**；`DT_NEEDED` 只依赖 `core.so.2.0.1`。用 `ctypes` 直接 `dlopen`（不初始化渲染器、不碰 GPU）确认
`LD_LIBRARY_PATH` 指向该目录时解析到的全是 2.0.1，共享目录分毫未动（soname 链接仍指向 2.3.3）。

这意味着决定性测试**不需要改共享环境**：只要在模拟器进程的 `LD_LIBRARY_PATH` 里换成这个目录即可。

## 1c. 2.0.1 在 Blackwell 上跑不了：09-28 的替换是必须的

直接解析两个 CUDA 设备库里嵌入的 fatbin（`cross-cluster/fatbin_targets.py`，本机没有 cuobjdump）：

| 库 | 原生 SASS 内核 | PTX |
| --- | --- | --- |
| `libOpenImageDenoise_device_cuda.so.2.0.1` | sm_70 sm_75 sm_80 sm_90 | **无** |
| `libOpenImageDenoise_device_cuda.so.2.3.3` | sm_70 sm_75 sm_80 sm_90 **sm_100 sm_120** | 无 |

本机 compute capability **12.0**。2.0.1 既没有 sm_120 内核、也没有可供驱动 JIT 的 PTX，**它的 CUDA 降噪在这张卡上无法运行**（wheel 里也没有 2.0.1 的 CPU 设备库可退）。2.3.3 的 tarball 于 09-28 04:32 下载、同日 04:33 建链——这不是误操作，是让 SAPIEN 光追降噪在 Blackwell 上跑起来的必要步骤。

**后果**：§1b 恢复的"干净 2.0.1"在这台机器上**不可用**，`run_official_oidn201.sh` 作废。Shenlong 没有任何办法用生成参考 PNG 的那个降噪器版本来回放场景。

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

~~改用 `LD_LIBRARY_PATH=openwam-runtime/oidn-2.0.1-pristine`~~ —— **作废**，见 §1c：2.0.1 没有 Blackwell 内核。这个检验只能在 Torch（L40S）上反向做，Torch 已完成（OIDN_AB_RESULT）。

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
