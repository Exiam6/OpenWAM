#!/bin/bash
# ex4b: takeover test for one arm on one GPU. Usage: run_takeover.sh <arm> <seed> <gpu_uuid> <pci> <port>
set -u
arm=$1 seed=$2 uuid=$3 pci=$4 port=$5
R=/home/zifanz4/openwam-runtime; E=$R/zero-task-ex4b/$arm-seed$seed; Z=/home/zifanz4/openwam-torch-dispatch/research/greene/zero-task
D=$R/rae-policy-expanded-20261006/data; mkdir -p $E
CK=$R/rae-factorial-20261010/runs/$arm-seed$seed/checkpoint_step_20000.safetensors
cd /home/zifanz4/openwam-rae-research
env -i HOME=/home/zifanz4 PATH=$R/policy-env/bin:/usr/bin:/bin PYTHONPATH=/home/zifanz4/openwam-rae-research CUDA_VISIBLE_DEVICES=$uuid \
  CUDA_HOME=$R/cuda-toolkit TORCH_CUDA_ARCH_LIST=12.0 TMPDIR=$R/tmp TORCH_EXTENSIONS_DIR=$R/cache/torch_extensions-sm120 TRITON_CACHE_DIR=$R/cache/triton-sm120 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
  $R/policy-env/bin/python -u $R/zero-task-ex4c/serve_policy.py --checkpoint $CK --port $port --log $E/rng.jsonl > $E/server.log 2>&1 &
SRV=$!
for i in $(seq 1 300); do grep -q ENDTOEND_SERVER_READY $E/server.log 2>/dev/null && break; kill -0 $SRV 2>/dev/null || { echo server died; tail $E/server.log; exit 1; }; sleep 3; done
env -i HOME=/home/zifanz4 USER=zifanz4 PATH=$R/benchmark-env/bin:$R/cuda-toolkit/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES=$uuid EXPECTED_RENDER_PCI=$pci \
  OMP_NUM_THREADS=4 CUDA_HOME=$R/cuda-toolkit TMPDIR=$R/tmp HF_HUB_OFFLINE=1 ROBOTWIN_PATH=$R/assets-source/benchmarks/RoboTwin \
  ROBOTWIN_RUNTIME_ROOT=$E/simulator-runtime ROBOTWIN_ENABLE_PLANNER_FALLBACK=1 WAM_ROOT=$R/assets-source \
  LD_LIBRARY_PATH=$R/benchmark-env/lib/python3.10/site-packages/sapien/oidn_library \
  timeout 7200 $R/benchmark-env/bin/python -u $Z/takeover.py $port $E/out 5 $D/handover_block-seed4101100 $D/handover_block-seed4101101 $D/handover_block-seed4101102 $D/handover_block-seed4101103 $D/handover_block-seed4101104 > $E/takeover.log 2>&1
echo "exit=$?"; kill $SRV 2>/dev/null; grep -a -o '{"id".*' $E/takeover.log
