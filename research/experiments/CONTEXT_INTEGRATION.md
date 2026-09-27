# Context recorder integration — CPU and loopback gates, 2026-09-22

NOT a GPU experiment, not launch-ready. Existing80 evaluations and all prior
protocols/results are unchanged. Implements the recording portion of the
unlaunched CONTEXT_CONTROL_DESIGN.md; no change to its fixed-command A/B contrast.

New ContextTrace records selected saved20D command and local fixed_command_index
separately from online_response. It reuses unchanged planner/physical getter
wrappers from AttributionTrace and the native eval/action conversion path.
No synthetic server_step or server_action is written for selected fixed commands.
Idle arm has online_response=null; shadow arm retains the complete actual reply.
A differing but valid online action is recorded, never executed or excluded.
Wrong counters/malformed replies are retained and abort before environment motion.
Initial request mismatches are saved before abort, with no prediction or motion.

Timings: rpc_elapsed_ns wraps the client predict_once call (including its own
serialization/transport overhead); null for idle. selection_elapsed_ns includes
local hashing and response copying. Neither is pure GPU compute latency. The old
ambiguous roundtrip_ns label is removed. Failed error logging cannot replace the
original transport exception; normal abort recording is still performed outside.

Validation:8 integration checks +6 updated transport regression checks passed.
The8 checks cover:
- Both modes through native eval/converter with fake physics/getters; exactly the
  saved16D targets and separate target versus measured-state fields.
- Actual pinned RoboTwin eval_policy loop compiled on CPU with fake environment:
  first scene400000, one setup/close, two resets, two get_obs, two actions; no
  expert re-filtering, and same ten-choice instruction RNG consumption.
- Wrong online counter saved then aborts before action.
- Initial request mismatch saved without prediction/action.
- Ambiguous timeout retains original exception and performs no action/retry.
- Existing record directory refused.
- Real WSPolicyClient over loopback to a STUB WebSocket server: exact ping/reset/
  obs counts and replies. No model policy is created.
- Real socket timeout: one obs sent, no retry or next command; own test server
  and socket close after the test. No other task/process is stopped.

Initial bootstrap attempt failed before tests because the private source checkout
has no runtime OpenWAM path. Corrected PYTHONPATH to the pinned runtime benchmark;
attempt-00-bootstrap.txt and subsequent full logs are retained. Attempt1 passed7
tests; adding a real socket-timeout gate made the final suite8, all passed. Source
format/lint checks pass. Test artifacts: results/context-integration-20260922.

Reproduce locally using the existing benchmark-env:
  PYTHONPATH=/data02/zifanz4/openwam-experiments/OpenWAM/benchmarks/robotwin \
    /data02/zifanz4/openwam-experiments/benchmark-env/bin/python \
    scripts/check_context_integration.py -v
  /data02/zifanz4/openwam-experiments/benchmark-env/bin/python \
    scripts/check_context_trial_transport.py -v
For another workspace, adjust pinned input/code paths in the test files. All
simulator-facing environment/getter objects in these CPU tests are substitutes;
passing is not evidence of real simulation determinism or model behavior.

Remaining before GPU: complete actual client entrypoint/lifecycle, fresh-process
model+sim runner, timeout/process-cleanup gates, complete runtime protocol/config/
source freeze, and a separate bounded resource window with fresh idle checks.
The expired23:45:59 allocation is not extended. This is preparation only; online
activity's influence on the original observation divergence is still unknown.
