# Updated recovery proposal: three zero-step seed44 failures

This supersedes the unexecuted02:30 proposal, which covered only PCA44 and S-VAE44. Wan44 also failed at03:09:02 before training began, with1MiB memory,100% sampled utilization and0ECC. All six seed42/43 runs completed exit0; complete per-seed batch streams match across all three methods.

The frozen pipeline plan has `no_restart_or_replay: true`; an explicit narrow exception remains pending. No approval file exists and this proposal has not been executed. Preparing it neither changes the scientific protocol nor revives expired budgets.

Exact plan: `preflight-recovery-20260924T0315.json`. Read-only by default: `/home/zifanz4/openwam-experiments/scripts/endtoend/recover_preflight_20260924T0315.py`. Execution additionally requires a separate explicit user-approval record bound to the exact plan/script hashes.

After approval: preserve all three original failures and controller logs in an attempt archive; check idle same-model L40S GPUs4/5/6 three times; launch each still-untrained seed44 job exactly once with unchanged model/data/config/source and only GPU placement metadata changed; resume the unchanged controller with remaining original wall time. All6000-update budgets, scenes, success criteria and analysis stay fixed. Any new failure stops recovery without automatic retry. Existing unrelated jobs remain untouched.
