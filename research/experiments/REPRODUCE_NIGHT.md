# Reproduce the minimal-contribution study

This package is an offline diagnostic study of the DINOv3 / S-VAE Study branch. It is not OpenWAM-alpha reproduction or closed-loop evaluation. Read NIGHT_PROTOCOL.md for the fixed tasks, seeds, episode split, probe definition, negative-result decision rule and numerical-control amendment.

## Environment and public assets

Use Python 3.10, one available NVIDIA GPU and sufficient storage (~15 GB of data/feature caches plus the Python environment). Follow your cluster's allocation rules. The measured run used one L40S; do not assume a GPU is free merely because it is visible.

```bash
python3.10 -m venv env
source env/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-lock.txt
git clone https://github.com/OpenWAM-Official/OpenWAM.git OpenWAM
git -C OpenWAM checkout 7c5861e45cfe1339a0323f0e0b03a3316c37971c
mkdir -p data
python scripts/download_pilot.py
python scripts/download_night.py
```

The downloaders use public revision-pinned assets and verify dataset SHA256 hashes. The DINO encoder weights are extracted from the released checkpoint via verified HTTP ranges; the complete WAM checkpoint is not downloaded. Feature extraction imports the unmodified upstream leaf modules without eager registration of unused WAM architectures.

## Execution

Set CUDA_VISIBLE_DEVICES to your allocated GPU. Every script uses only its visible device 0. Keep the generated files under this package root; paths are resolved relative to the scripts.

```bash
export CUDA_VISIBLE_DEVICES=YOUR_ALLOCATED_GPU
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
python scripts/pilot.py --stage extract
python scripts/night.py --check-only
python scripts/night.py
python scripts/correct_corruptions.py
python scripts/summarize_night.py
```

All 18 reducers train for a fixed 2,000 steps. Final test analysis uses `evaluation-matched-batch` exclusively: the initial single-frame corruption encoding is preserved only as an audit trail. The correction checks clean-frame agreement for all 360 held-out clips before accepting perturbation results. The published figures summarize all three seeds and all three tasks.

The experiment scripts are separate from the proposed upstream contribution. The contribution patch adds only a standalone evaluator, tests, documentation and a README link; it does not alter model training or inference. Full upstream dependency-suite validation remains necessary before submitting a PR.
