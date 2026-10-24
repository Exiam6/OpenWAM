# Native RoboTwin evaluation protocol

Run this on an RTX/Vulkan RT host after the policy and RoboTwin environments,
source checkout, model assets, and checkpoint files have been copied there.
The current eight-A100 host has no working NVIDIA Vulkan ICD and cannot run
the native renderer. CPU Lavapipe outcomes remain a separate diagnostic.

## Frozen comparison

- Evaluate the 10 tasks in `research/rae/DATA_SELECTION.md` with `demo_clean`,
  unseen instructions, and 100 successful expert-feasibility seeds per task.
- Build one sealed RoboTwin manifest per task with the native RTX renderer.
  Reuse **the same manifest hash** for all six matched checkpoints: native
  VAE and DINOv2 RAE at training seeds 42, 43, and 44.
- Use `checkpoint_step_1000.safetensors` for every matched checkpoint. Verify
  each file against its registered SHA256 and the saved source commit marker.
- Evaluate the released RoboTwin-Full checkpoint separately on the complete
  official `demo_clean` and `demo_randomized` benchmark: 50 tasks per mode,
  100 episodes per task. Its training/data provenance differs from the matched
  Foundation-initialized experiment; report it as a reproduction reference.
- Keep `evaluation_video_shift=5`, the released three-camera order, and the
  registered policy inference settings fixed across each matched pair.

The existing `benchmarks/robotwin/labtasker_submit.py` and
`labtasker_worker.py` implement sealed manifest construction, exact episode
selection, bounded batch retries, and structured per-episode results. Install
`benchmarks/robotwin/requirements-labtasker.txt` in the **RTX host's** OpenWAM
policy environment and follow `benchmarks/robotwin/LABTASKER.md`. Keep the
manifest cache, Labtasker database, and run logs on persistent storage.

For the selected 10 tasks, submit the manifest build with:

```bash
python benchmarks/robotwin/labtasker_submit.py --operation build_manifest \
  --mode demo_clean --tasks beat_block_hammer,hanging_mug,open_microwave,place_bread_skillet,place_burger_fries,place_dual_shoes,shake_bottle_horizontally,stack_blocks_three,stack_bowls_three,stack_bowls_two \
  --episodes 100 --manifest-dir /path/to/persistent/manifest-cache
```

After all manifest Tasks succeed, submit one `run_eval` job per checkpoint
with the same `--mode`, `--tasks`, `--episodes 100`,
`--episode-batch-size 5`, `--instruction-type unseen`, and
`--manifest-dir`; put `--ckpt-dir /path/to/checkpoint` after `--`. Run
workers on the RTX device(s) and verify that the selected Vulkan renderer is
the intended NVIDIA GPU. The native path must not set `ROBOTWIN_SOFTWARE_RT`,
`ROBOTWIN_SOFTWARE_RASTER`, or a Lavapipe `VK_ICD_FILENAMES`.

## Acceptance and reporting

1. Require all expected Labtasker batches to succeed. Technical failures are
   logged and retried; do not count them as robot policy failures.
2. Verify each evaluation Task's manifest hash, task, mode, instruction type,
   episode range, and checkpoint identity. Require exactly 100 distinct matched
   episode indices and accepted scene seeds per task and per checkpoint.
3. Report per-task and task-balanced success rates, plus the 3-seed matched
   VAE/RAE difference. Bootstrap the paired difference over training seeds,
   tasks, and episode indices; also give discordant-pair counts.
4. Report the released-checkpoint reproduction separately, with coverage and
   technical failures for both modes. Do not mix its episodes into the causal
   comparison.
5. Preserve each checkpoint SHA256, source commit, manifest SHA, full Task
   results, simulator logs, and summary tables with the final report.

Current offline action loss and CPU-rendered episodes do not satisfy this
native success-rate gate.
