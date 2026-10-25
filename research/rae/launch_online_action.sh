#!/usr/bin/env bash
# Seed-42 online action-grounded DINO ablation; called only after the registered queue.
set -euo pipefail

steps="${1:?usage: launch_online_action.sh STEPS OUTPUT_DIR}"
output="${2:?usage: launch_online_action.sh STEPS OUTPUT_DIR}"
if [[ "$steps" != 2 && "$steps" != 1000 ]]; then
  echo "steps must be 2 (integration smoke) or 1000 (experiment)" >&2
  exit 2
fi

repo="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "Commit the isolated worktree before training" >&2
  exit 1
fi
graft=/data/zifan-openwam-rae/online_action_adapter_20261025/graft_seed42
data=/root/zifan/openwam-assets/robotwin-splits/train/dataset
if [[ ! -f "$graft/graft_manifest.json" || ! -f "$data/../../split_manifest.json" ]]; then
  echo "Registered graft or train split is missing" >&2
  exit 1
fi

mkdir -p "$output"
source_commit="$(git -C "$repo" rev-parse HEAD)"
marker="$output/source_commit_at_launch.txt"
if [[ -f "$marker" ]]; then
  [[ "$(cat "$marker")" == "$source_commit" ]] || { echo "Output belongs to another source commit" >&2; exit 1; }
else
  printf '%s\n' "$source_commit" > "$marker"
fi

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export PATH=/root/zifan/openwam-envs/policy/bin:"$PATH"
export CUDA_HOME=/root/zifan/openwam-envs/benchmark-conda
export TORCH_CUDA_ARCH_LIST=8.0
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export NCCL_RAS_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NPROC_PER_NODE=8
export MASTER_PORT=29695
export OMP_NUM_THREADS=4

cd "$repo"
exec bash scripts/train.sh \
  dataloader=robotwin \
  "dataloader.dataset_dir=$data" \
  dataloader.variant=clean_50 \
  dataloader.split=train \
  training.num_epochs=null \
  "training.max_steps=$steps" \
  training.batch_size=1 \
  training.gradient_accumulation_steps=1 \
  training.dataset_num_workers=2 \
  training.learning_rate=1e-5 \
  training.action_lr=1e-4 \
  training.video_lr=1e-5 \
  training.initialize_model_on_cpu=true \
  training.save_steps=2000 \
  training.save_full_states_for_resume=false \
  "training.finetune_ckpt_path=$graft" \
  training.reset_visual_projections=false \
  "training.output_path=$output" \
  project.seed=42 \
  project.wandb.project=null \
  model/video_backbone/encoder=dinov2 \
  model.video_backbone.from_scratch=true \
  model.video_backbone.encoder.model_path=/root/zifan/openwam-assets/dinov2-base \
  model.video_backbone.encoder.feature_stats_path=/root/zifan/openwam-runs/rae_feature_stats_stride16.pt \
  model.video_backbone.encoder.online_action_adapter_hidden_dim=128 \
  'model.freeze=[video_backbone.text_encoder,video_backbone.reason1,video_backbone.vae,video_backbone.image_encoder]'
