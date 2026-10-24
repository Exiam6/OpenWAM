# Native RoboTwin evaluation protocol

Run this on an RTX/Vulkan RT host after the policy and RoboTwin environments,
source checkout, model assets, and **entire checkpoint directories** have been
copied there. Each directory includes `config.yaml`, tokenizer files, and, for
RAE, the `dinov2/config.json` sidecar in addition to the `.safetensors` file.
The current eight-A100 host has no working NVIDIA Vulkan ICD and cannot run
the native renderer. CPU Lavapipe outcomes remain a separate diagnostic.

The current host's optional Tailscale userspace client exposes a local SOCKS5
listener at `127.0.0.1:1055`. After that client has joined Hestia's tailnet,
use `ssh -o 'ProxyCommand=python research/rae/socks5_ssh_proxy.py %h %p'`
with Hestia's SSH user, key, and Tailscale IP to access the RTX host. This
proxy leaves the A100 host's system routes unchanged. Verify the SSH host key
and remote GPU before transferring checkpoints or starting Workers.

## Frozen comparison

- Evaluate the 10 tasks in `research/rae/DATA_SELECTION.md` with `demo_clean`,
  unseen instructions, and 100 successful expert-feasibility seeds per task.
- Build one sealed RoboTwin manifest per task with the native RTX renderer.
  Reuse **the same manifest hash** for all six matched checkpoints: native
  VAE and DINOv2 RAE at training seeds 42, 43, and 44.
- The original 50 demonstrations per selected task have scene seeds in
  `[0, 167]`; `seed.txt` maps episode order to those seeds. The frozen
  train/dev/test split has no episode overlap. RoboTwin's evaluator starts
  candidate scene seeds at `100000`, so its native manifest is scene-disjoint
  from these demonstrations when that evaluator rule is preserved. The
  per-task seed-file hashes and split mapping are recorded in
  `/root/zifan/openwam-runs/robotwin_native_seed_overlap_audit.json`
  (SHA256 `7fef743052e3ac59155d471f0f0b6d66632c4814a01aa0f824e7009529af3134`).
  The [OpenWAM paper, Table 4](https://arxiv.org/pdf/2609.07398) lists the
  Foundation donor's five pretraining sources as egocentric video,
  AgiBotWorld-Beta, RoboCOIN, DROID, and InternData-A1; RoboTwin is not a
  listed source. That source-level list does not prove that every visual
  scene or asset in the native evaluation is absent from pretraining.
- Use `checkpoint_step_1000.safetensors` for every matched checkpoint. Verify
  each file against its registered SHA256. Seed 42 predates the automatic
  launch marker: its RAE and VAE source commits are recorded in the respective
  `wam_pilot_20261023/{rae,vae}/registration.json` files (the VAE registration
  also records the shared model-code commit). For seeds 43 and 44, verify
  `source_commit_at_launch.txt` in each run root.
  The registered seed-42 model-code commit is `448ee84d`; the later matched
  source commit is `c414e853`. Their `openwam/`, `scripts/`, and `configs/`
  diff changes S-VAE collection/training/reducer files, not the native VAE or
  raw-DINOv2 forward path. The pilot launcher changes add seed/path overrides,
  S-VAE opt-in, and provenance checks; the native VAE and raw-DINOv2 training
  settings remain the same. Keep this source-version distinction in the report.
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

For the released-checkpoint reproduction, build separate manifests for the
full 50-task `demo_clean` and `demo_randomized` modes, each with `--tasks all`
and `--episodes 100`. Submit one `run_eval` job per mode against the released
RoboTwin-Full checkpoint, reusing its mode's manifest directory. This produces
10,000 attempted episodes in total. Keep these submission IDs separate from
the six matched-checkpoint submissions.

After both official submissions finish, create a registry JSON with keys
`"demo_clean"` and `"demo_randomized"`. Each entry contains its
`submission_id`, the absolute `checkpoint` file path, and its
`checkpoint_sha256`; both entries must identify the same released checkpoint.
Then run:

```bash
python research/rae/summarize_native_official.py \
  --registry /path/to/official-submissions.json \
  --output /path/to/native-official-summary.json
```

This audit requires every one of the 50 tasks and 100 distinct episodes per
task in each mode, verifies the sealed manifest and checkpoint, and refuses
partial or technically failed submissions before reporting success rates.

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

After all six matched submissions finish, create a registry JSON with keys
`"42"`, `"43"`, and `"44"`. Each has `"vae"` and `"rae"` entries containing
`submission_id`, the absolute `checkpoint` file path, and
`checkpoint_sha256`. Then run:

```bash
python research/rae/summarize_native_paired.py \
  --registry /path/to/paired-submissions.json \
  --output /path/to/native-paired-summary.json
```

The summarizer rejects incomplete batches, checkpoint hash changes, duplicated
episodes, manifest or scene mismatches, and incorrect task coverage before
calculating the three-seed paired success delta and 20,000-draw interval.

Current offline action loss and CPU-rendered episodes do not satisfy this
native success-rate gate.
