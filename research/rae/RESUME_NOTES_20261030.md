# Recovery preparation — 2026-10-30

The user has indicated RTX6000 resources are available and requested pushing
progress before continuing. The concrete host/allocation still needs to be
identified. The earlier local A100 prohibition and stopped Hestia jobs must
not be treated as automatically lifted by the availability of another GPU.
No GPU training or evaluation was launched in this preparation.

## Fixed recovery of partially completed matched seeds

`run_additional_seeds.py` previously tried to rebuild the seed's temporary
graft and launch RAE training even when a completed RAE checkpoint already
existed. It also unconditionally reran offline evaluations into existing log
paths, which would fail rather than resume seed 44 after its completed RAE
training/evaluation.

The controller now:

- Checks for completed checkpoints before creating a graft or launching any
  training, verifying the final-save marker, config seed/encoder, safetensors
  header, byte size, and SHA256 before reuse.
- Reuses the completed seed-44 RAE and launches only its missing VAE run;
  the deleted temporary RAE graft is not required in this path.
- Validates saved offline evaluations against checkpoint path/hash, split,
  evaluation seed, coverage, repeat counts, and action-loss aggregates.
- Generates missing paired comparisons from the audited reports and checks
  existing comparisons reproduce exactly, without rerunning completed GPU
  evaluations.
- Refuses to overwrite an unfinished training attempt or a conflicting
  evaluation/comparison. No optimizer-state continuation is implied; these
  pilot checkpoints still contain weights rather than full optimizer state.

Seven recovery tests plus ten native-runner/checkpoint-registry tests passed
on CPU. Ten existing offline dev/test reports passed the reuse validator;
all four existing seed-42/43 paired comparisons regenerated identically.
These tests establish controller recovery behavior, not model performance.
The original dev-only evaluator omitted the `split` field. Its historical
meaning is preserved as dev; such a report is explicitly rejected for test
reuse.

The original processes already loaded their old Python code. Do not simply
send `SIGCONT` to the old queue. After resource authorization and baseline
checks, run the updated controller from this branch while preserving the
frozen training worktree and existing artifacts/source markers.

## Native baseline ordering still requires a separate route

The stopped production queue retains 144 pending seed-42 RAE batches.
Submitting official-model diagnostic Tasks to the same route would let
workers claim those older batches first. The official diagnostic needs a
distinct Labtasker route, submission ID, and output directory, while reusing
the sealed scene manifests. This diagnostic is preparation for the full
registered evaluations and does not count as their completion. Its GPU
execution has not been launched.

The implementation is now
[`run_official_native_probe.py`](run_official_native_probe.py). It defaults
to a read-only plan for three tasks and ten episodes per task (six batches,
30 episodes). CPU tests check private routing, stable resubmission identity,
missing/incomplete coverage rejection, wrong-model/scene/instruction rejection,
technical-versus-robot outcome handling, and Worker cleanup on timeout.
Together with the native-runner and seed-recovery checks, 23 CPU tests passed.
A dry run against the registered official checkpoint metadata also confirmed
30 planned episodes with no runtime directory created or processes launched.
This does not establish that the simulator/model can run on an untested host.

After an RTX host and GPU index have been authorized, run the following from
its OpenWAM checkout, adjusting the host-local paths when migrating:

```bash
../envs/orchestrator/bin/python -m research.rae.run_official_native_probe \
  --checkpoints ../native-eval/production/checkpoint_registry.json \
  --manifest-dir ../native-eval/production/manifests \
  --run-dir ../native-eval/official-healthcheck-20261030 \
  --robotwin-root ../RoboTwin \
  --robotwin-python ../envs/simulator/bin/python \
  --policy-python ../envs/policy/bin/python \
  --gpu APPROVED_GPU_INDEX
```

The command above prints the plan. Add `--execute` to run it on authorized
resources. Execution verifies the released checkpoint bytes and selected RTX
device and runs the existing native-host preflight. It does not start or
resume any training or production Worker. Accepted diagnostic results and
videos live in the separate run directory; the three tasks are
`stack_blocks_three`, `open_microwave`, and `shake_bottle_horizontally`.

Exit 2 with `summary.json` and `all_zero_success: true` means all diagnostic
robot attempts failed even though infrastructure Tasks completed. An
infrastructure failure raises an error instead. Neither exit 0 nor a single
success establishes the full official benchmark; inspect per-task outcomes
and videos before deciding how to continue the complete registered queue.

Current evidence and remaining full scope are recorded in
[`PROGRESS_AUDIT_20261030.md`](PROGRESS_AUDIT_20261030.md).
