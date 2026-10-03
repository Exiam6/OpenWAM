# Torch 环境清单（cross-cluster-rootcause-20261003 §1）

执行方：NYU Torch。任务说明：[TORCH_TASK_20261003.md](TORCH_TASK_20261003.md)。
原始记录：[records/torch-env-inventory-20261003.json](records/torch-env-inventory-20261003.json)，
由 job **19112609**（L40S，gl037，2026-10-03 13:45 EDT，45 s，未跑策略、未计分）用
`cross-cluster/env_inventory.py` 采集，环境与 `sbatch/official_baseline.sbatch` 相同（bench env，unset PYTHONPATH）。
脚本里做了一次极小的 `rt` + OIDN 渲染（参数同 RoboTwin `envs/_base_task.py:214-217`），
目的是从 `/proc/self/maps` 读出**实际加载**的库。

| 项 | Torch | Shenlong（任务说明所列） |
| --- | --- | --- |
| GPU / CC | NVIDIA L40S / **sm_89** | Blackwell / sm_120 |
| NVIDIA 驱动 | **610.43.02**（CUDA UMD 13.3），见下方注意 | 580.95.05 |
| policy env torch | 2.7.0+cu126，cuDNN 90501 | — |
| bench env torch | 2.4.1+cu124 | — |
| nvcc（CUDA_HOME） | 12.4.131 | — |
| SAPIEN | 3.0.0b1 | — |
| **OIDN 实际加载** | **SAPIEN 自带 `sapien/oidn_library/libOpenImageDenoise.so.2.0.1`**（+ core、device_cuda 2.0.1） | `LD_LIBRARY_PATH` 指向 oidn_library；`downloads/` 下另有 oidn-2.3.3 |
| `LD_LIBRARY_PATH` | **未设置** | 已设置 |
| `SAPIEN_*` / `VK_*` 环境变量 | 无 | — |
| 渲染后端 | 光追：shader `rt`，spp 32，path depth 8，denoiser `oidn`（RoboTwin 代码设定）；设备 L40S rayTrace=1 | — |
| Vulkan | `libvulkan.so.1.4.328`，`libnvidia-rtcore/glvkspirv 610.43.02`；系统 nvidia ICD 没被找到，SAPIEN 用了自己提供的 ICD（同样的 UserWarning 也出现在 18771822 的全部 4 份 run.log 里，所以和基线一致） | — |
| **`ROBOTWIN_ENABLE_PLANNER_FALLBACK`** | **job 18771822 设置了 `=1`**：`baseline-official/scripts/run_baseline.py` 写死 `'1'`，且当前文件 sha256 `43becf07…` 与 18771822 的 run-record 一致 | `=1` |

## 结论

1. **planner fallback 可以排除**：两边都是 `=1`。
2. **OIDN 是一个具体的待查差异**：Torch 实际跑的是 SAPIEN 3.0.0b1 自带的 **OIDN 2.0.1**。请 Shenlong 用同样的
   `/proc/self/maps` 方法确认那边实际加载哪个版本（2.0.1 还是 2.3.3）。另外，OIDN 2.0.1 是在 Blackwell 出现之前发布的，
   它在 sm_120 上走的是 CUDA 设备还是 CPU 回退，也值得一起记下来。光追降噪会直接改变像素，
   这与 Shenlong 那边 MAE 0.51–1.03 的整体偏移相符。
3. **驱动注意**：18771822 当时（2026-09-29，gl029）没有记录驱动版本，19112609（gl037）读到的是 610.43.02。
   不能排除集群在这几天内升过驱动，所以"18771822 跑在 610.43.02 上"这一点**未经证实**。
   如果需要严格对应，§2 的渲染测量在当前驱动上跑，可以同时给出当前驱动下的参照值。

## §2 状态

**阻塞**：`render_survey.py` / `render_sens_all.py` / `action_sensitivity.py` 不在
`research/progress-20260927` 分支上，也不在 Torch 运行时目录里。推上来后（请注明路径和需要的环境变量），
Torch 按任务说明 §2 原样运行，不改闸门、不改 serving 配置、不重跑基线。
