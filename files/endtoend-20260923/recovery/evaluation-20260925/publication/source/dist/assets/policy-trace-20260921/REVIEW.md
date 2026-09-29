# Draft contribution: optional policy request/action tracing

Status: local reviewable patch; no upstream PR or maintainer message sent.
Base: 7c5861e45cfe1339a0323f0e0b03a3316c37971c
Contribution: d1273ca685d7f65b3754843b5fe42a210ecdcfa0
Patch SHA256: da0f13c5b4ffaf79325e163c35836486667dfadbc3eec53b37574102a7ab2c52

Repeated runs can agree on task success while observations diverge before the
next buffered action chunk. An optional wrapper records submitted request,
image-payload, prompt and state hashes, returned server actions and step counters.
The comparator reports each field's first differing event, lengths and metadata.
Reset and error events count as events. No default evaluation or training changes.

Four files, 363 added lines including documentation and tests. Runtime utility:
benchmarks/utils/policy_trace.py. Usage: benchmarks/POLICY_TRACE.md.

Validation: 6 unit tests including actual RoboTwin eval 20D-to-16D action parity;
Ruff and diff checks passed. Patch applies to the pinned upstream base. CPU-only
saved-response replay processed all 18 existing traces / 1881 requests; all 1899
source-file hashes remained unchanged. It recovered request/image/state index1
and action index32 differences in all 6 original closed-loop pairs, and equality
in all 3 short fixed-command pairs. This is software validation using old data,
not new independent evaluation or live WebSocket validation.

Scope/limits: closes its underlying client; use one wrapper per synchronous loop.
Log/serialization failures propagate; tracing adds I/O and timing overhead.
Recorded actions are server responses, not measured joint motion. Native RoboTwin
joint_action.vector is drive targets. Encoded image hashes are not pixel metrics.
Finite trace equality proves neither global determinism nor a fixed root cause.
No claim of RAE superiority or closed-loop representation improvement.

Apply in a clean checkout of the pinned base:
  git apply --check policy-trace.patch
  git apply policy-trace.patch
  python -m unittest tests.test_policy_trace -v
  python -m benchmarks.utils.policy_trace repeat0.jsonl repeat1.jsonl

Use the existing thin-client dependencies. The full replay script reads the
original saved request gzip files; adjust SOURCE/CODE/OUT for another workspace.
The public trace archive includes generated lightweight logs, not all raw images.
