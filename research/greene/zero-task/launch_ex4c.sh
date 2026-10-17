#!/bin/bash
# ex4c: 6 cells (raw768_native, svae48_native x seeds 42/43/44) on gpu-01 GPUs 1-4, two per GPU.
set -u
R=/home/zifanz4/openwam-runtime; E=$R/zero-task-ex4c
declare -A UUID=([1]=GPU-89770bc8-d310-99d8-b3a4-d2b14c93d1a9 [2]=GPU-5588ad46-6e71-bb2c-8f18-80b2e7a14b5e [3]=GPU-6cddff6e-dafd-6282-972c-024127fe311c [4]=GPU-4076aefe-7b64-d6e7-8069-d8417062412b)
declare -A PCI=([1]=00000000:07:00.0 [2]=00000000:75:00.0 [3]=00000000:76:00.0 [4]=00000000:84:00.0)
cells=(raw768_native:42:1:1 raw768_native:43:2:2 raw768_native:44:3:3 svae48_native:42:4:4 svae48_native:43:1:9 svae48_native:44:2:10)
for c in "${cells[@]}"; do IFS=: read v s g vid <<< "$c"
  setsid nohup env -i HOME=/home/zifanz4 USER=zifanz4 PATH=$R/policy-env/bin:$R/cuda-toolkit/bin:/usr/bin:/bin \
    CUDA_VISIBLE_DEVICES=${UUID[$g]} STUDY_GPU=$vid EXPECTED_RENDER_PCI=${PCI[$g]} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    CUDA_HOME=$R/cuda-toolkit CPATH=$R/cuda-toolkit/targets/x86_64-linux/include TORCH_CUDA_ARCH_LIST=12.0 TMPDIR=$R/tmp MAX_JOBS=2 \
    TORCH_EXTENSIONS_DIR=$R/cache/torch_extensions-sm120 TRITON_CACHE_DIR=$R/cache/triton-sm120 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled ROBOTWIN_PATH=$R/assets-source/benchmarks/RoboTwin ROBOTWIN_ENABLE_PLANNER_FALLBACK=1 \
    WAM_ROOT=$R/assets-source LD_LIBRARY_PATH=$R/benchmark-env/lib/python3.10/site-packages/sapien/oidn_library PYTHONPATH=/home/zifanz4/openwam-rae-research \
    EX4C_TASKS=handover_block,adjust_bottle \
    $R/policy-env/bin/python -u $E/evaluate_cell.py --variant $v --seed $s > $E/cell-$v-seed$s.log 2>&1 < /dev/null &
  echo "$v seed$s -> GPU$g (port id $vid) pid $!"
done
