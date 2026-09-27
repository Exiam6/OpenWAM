# Reproduce group-supervision follow-up

Read GROUP_PROTOCOL.md before interpreting. The study reuses the deterministic control study's old-task features, PCA training statistics, vectors, and lambda=0/1 final checkpoints. It never reevaluates the old test labels. Unchanged source modules: scripts/control_study.py, scripts/pilot.py, scripts/night.py. The additional experiment driver is scripts/group_study.py.

## Inputs and execution
1. Prepare prior datasets/features/checkpoints following REPRODUCE_NIGHT.md and REPRODUCE_CONTROL.md. Preserve their relative results/night-20260921 and results/control-20260921 paths. The group driver requires all three old tasks and three seeds.
2. Download the fresh pinned task using `python scripts/download_group.py`; checksum and archive traversal validation are mandatory.
3. With one healthy idle GPU, export its UUID as CUDA_VISIBLE_DEVICES, CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP_NUM_THREADS=6, OPENBLAS_NUM_THREADS=6, MKL_NUM_THREADS=6. Run `env/bin/python -u scripts/group_study.py`. The driver checks for foreign GPU processes and uncorrected ECC errors.
4. Check reuse-audit.json requires max parameter difference exactly zero. If nonzero, the study aborts instead of mixing unmatched controls.
5. selection.json is saved after the 18 development runs and before any fresh-task training/test evaluation. test-unlock.json records its hash after all 9 fresh runs; summary.json and complete.json are written at completion.

The actual execution moved from Group2 to cm001 GPU0 because the initially free Group2 device was claimed by another process twice before our GPU initialization. Both launches were stopped by the preflight; their logs remain on Group2. Existing venv packages and selected experiment inputs were copied to /data02/zifanz4/openwam-experiments. The runtime Python entrypoint remains the same system Python 3.10. No old environment or source driver was modified. The new full-length numerical audit establishes the original control checkpoint is reproduced exactly on this node.

## Separate policy work
scripts/download_policy.py retrieves the complete revision-pinned published DINO/S-VAE Study checkpoint and self-contained artifacts, validating LFS SHA256 values. policy-env is an isolated venv whose experiment-base.pth exposes the copied read-only experiment environment packages; extra inference packages are installed only in policy-env. policy-requirements-lock.txt records the resolved versions. This environment does not use the fake-package leaf bootstrap used by the compression experiments.

scripts/policy_smoke.py uses the official server construction and observation preprocessing paths in-process, with two real observations from a training episode. No actions are executed. Ten synchronous denoising steps, DiT cache enabled, compile disabled for setup diagnosis. This differs from the official compiled deployment protocol and must not be reported as a benchmark success rate.

RoboTwin is a separate checkout at 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4 with a separate benchmark-env (torch2.4.1+cu124). scripts/download_benchmark.py retrieves the revision-pinned robot/object assets for clean-mode evaluation, without the unused 11GB randomized background archive. Asset downloads and environment preparation alone do not establish simulator or closed-loop functionality.

## Completed execution evidence
The group study completed28training runs with exit0. Selected rho=.5, frozen before fresh-task fitting; selection.json's SHA256 equals test-unlock.json. All five recorded source hashes still match. Test metrics and confidence intervals are in summary.json.

The published-policy smoke completed5/5 clean episodes using the official adapter, seed argument0 (actual episode seeds100000–100004), original400-step limit, no planner fallback, compile disabled. The runtime preserved the original upstream code. See POLICY_SMOKE_PROTOCOL.md and the five videos. The original result file says.05 due to a verified denominator bug; our sidecar uses actual rollout numerator/denominator.

CPU reproduction of that independent integration bug, without a simulator/GPU:
```bash
python scripts/reproduce_smoke_denominator.py --robotwin /path/to/RoboTwin --openwam /path/to/OpenWAM
```
This loads only the reporting function and actual OpenWAM episode-cap wrapper, with stub successful rollouts. It emits a minimal RoboTwin patch and checks capped5/default100/emptyoverride outcomes. It never edits either supplied checkout.
