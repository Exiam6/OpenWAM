# Prepared recovery; not executed

Four completed runs are preserved. PCA44 and S-VAE44 stopped before training started: final idle utilization samples were86% and80% despite1MiB memory; current repeated checks are0% with no compute process. A stale sampling window after the previous job is the leading explanation, not a proven driver fault.

The frozen pipeline plan explicitly sets `no_restart_or_replay: true`, and the user asked to preserve the frozen protocol. This proposed exception is limited to these two zero-step launch failures and the dependent controller. It requires explicit user approval; the prepared script defaults to read-only checks and an execution flag additionally requires a separate approval record bound to the exact plan/script hashes.

The exact plan is `preflight-recovery-20260924T0230.json`; runner is `/home/zifanz4/openwam-experiments/scripts/endtoend/recover_preflight_20260924.py`. It archives every original failure, rechecks GPU idleness, launches the two unchanged jobs once, and resumes the unchanged controller within its original deadline. Existing Wan workers, all completed models, data, metrics, statistical criteria and all scientific protocol hashes remain unchanged. Any new failure invalidates this plan and is not covered by this proposed exception.

No recovery or new GPU work has been launched by preparing this plan.

Resource revision: GPUs1/2 were taken by another user at02:34:33. The exact prepared plan now selects currently idle cm009 L40S GPU4 for S-VAE44 and GPU5 for PCA44, subject to fresh checks. Only resource-placement metadata changes in separate retry job copies; original job/config/source files stay intact. The first read-only readiness check correctly rejected occupied slots; no recovery was executed.
