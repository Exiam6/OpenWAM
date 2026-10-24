#!/usr/bin/env bash
# Matched single-seed VAE / calibrated-DINOv2 WAM pilot on the frozen ten-task split.
set -euo pipefail

variant="${1:?usage: launch_wam_pilot.sh vae|rae}"
if [[ "$variant" != vae && "$variant" != rae ]]; then
  echo "variant must be vae or rae" >&2
  exit 2
fi

repo="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "Commit research changes before launching training" >&2
  exit 1
fi

foundation=/root/zifan/openwam-assets/foundation-checkpoint
seed="${WAM_PILOT_SEED:-42}"
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "WAM_PILOT_SEED must be a non-negative integer" >&2
  exit 2
fi
graft="${WAM_PILOT_GRAFT:-/root/zifan/openwam-runs/rae_foundation_graft_calibrated_seed${seed}}"
svae_path="${WAM_PILOT_RAE_SVAE_PATH:-}"
data=/root/zifan/openwam-assets/robotwin-splits/train/dataset
run_root="${WAM_PILOT_RUN_ROOT:-/root/zifan/openwam-runs/wam_pilot_20261023}"

if [[ ! -f "$data/../../split_manifest.json" ]]; then
  echo "Missing episode split manifest" >&2
  exit 1
fi
if [[ "$variant" == vae ]]; then
  source="$foundation"
  reset_projections=true
  encoder_overrides=()
else
  source="$graft"
  reset_projections=false
  encoder_overrides=(
    model/video_backbone/encoder=dinov2
    model.video_backbone.from_scratch=true
    model.video_backbone.encoder.model_path=/root/zifan/openwam-assets/dinov2-base
    model.video_backbone.encoder.feature_stats_path=/root/zifan/openwam-runs/rae_feature_stats_stride16.pt
  )
  if [[ -n "$svae_path" ]]; then
    if [[ ! -f "$svae_path" ]]; then
      echo "Missing S-VAE reducer: $svae_path" >&2
      exit 1
    fi
    encoder_overrides=(
      model/video_backbone/encoder=dinov2
      model.video_backbone.from_scratch=true
      model.video_backbone.encoder.model_path=/root/zifan/openwam-assets/dinov2-base
      model.video_backbone.encoder.feature_stats_path=null
      "model.video_backbone.encoder.svae_path=$svae_path"
      model.video_backbone.encoder.svae_target_dim=48
    )
  fi
fi
if [[ ! -f "$source/config.yaml" || ! -f "$source/graft_manifest.json" && "$variant" == rae ]]; then
  echo "Missing checkpoint source or graft manifest: $source" >&2
  exit 1
fi
if [[ "$variant" == rae ]]; then
  /root/zifan/openwam-envs/policy/bin/python - "$source/graft_manifest.json" "$seed" "$svae_path" <<'PY'
import hashlib
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("seed") != int(sys.argv[2]):
    raise SystemExit(f"Graft seed {manifest.get('seed')} does not match run seed {sys.argv[2]}")
expected = manifest.get("svae_sha256")
if sys.argv[3]:
    digest = hashlib.sha256()
    with open(sys.argv[3], "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected or manifest.get("svae_latent_dim") != 48:
        raise SystemExit("Graft S-VAE reducer hash or target width does not match the requested run")
elif expected is not None:
    raise SystemExit("Graft includes an S-VAE reducer but the run did not request it")
PY
fi

export PATH=/root/zifan/openwam-envs/policy/bin:"$PATH"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/root/zifan/openwam-envs/benchmark-conda
export TORCH_CUDA_ARCH_LIST=8.0
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export NCCL_RAS_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NPROC_PER_NODE=8
export MASTER_PORT="${MASTER_PORT:-29617}"
export OMP_NUM_THREADS=4

cd "$repo"
source_commit="$(git rev-parse HEAD)"
source_marker="$run_root/$variant/source_commit_at_launch.txt"
mkdir -p "$(dirname "$source_marker")"
if [[ -f "$source_marker" ]]; then
  if [[ "$(cat "$source_marker")" != "$source_commit" ]]; then
    echo "Training output already registered to a different source commit: $source_marker" >&2
    exit 1
  fi
else
  printf '%s\n' "$source_commit" > "$source_marker"
fi
# OpenWAM only writes final weights when save_steps is truthy; 2000 > max_steps
# gives one final checkpoint at step 1000 without periodic checkpoint disk use.
exec bash scripts/train.sh \
  dataloader=robotwin \
  "dataloader.dataset_dir=$data" \
  dataloader.variant=clean_50 \
  dataloader.split=train \
  training.num_epochs=null \
  training.max_steps=1000 \
  training.batch_size=1 \
  training.gradient_accumulation_steps=1 \
  training.dataset_num_workers=2 \
  training.learning_rate=1e-5 \
  training.action_lr=1e-4 \
  training.video_lr=1e-5 \
  training.initialize_model_on_cpu=true \
  training.save_steps=2000 \
  training.save_full_states_for_resume=false \
  "training.finetune_ckpt_path=$source" \
  "training.reset_visual_projections=$reset_projections" \
  "training.output_path=$run_root/$variant" \
  "project.seed=$seed" \
  project.wandb.project=null \
  "${encoder_overrides[@]}"
