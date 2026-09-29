# Resource-only resumption amendment — 2026-09-21 17:27 CDT

The original study completed all identity, renorm and adapter conditions (60
primary + 3 historical + 10 feasibility episodes) and stopped before loading the
final adapter_renorm server. Its GPU2 preflight required memory<100 MiB and 0%
utilization; a same-user G1 simulation using GPU1 also held a GPU2 graphics context.
At diagnosis that context used233 MiB, with no GPU2 compute process and zero ECC.
The original failed exit code, launcher and audit logs are preserved in
interruption-20260921T1627. No incomplete final-arm episode exists.

Resume only adapter_renorm on the SAME GPU2. Do not stop or change the other task.
The revised preflight allows at most ONE graphics-only context, owned by this
UID and identified by exact G1 Python executable and terrain entrypoint. Its
memory must be<=384 MiB; total GPU memory<=512 MiB, utilization<=15%, ECC=0,
three consecutive samples five seconds apart. Any compute/unknown context fails.
Original flock and localhost port checks remain. Startup patience increases
from120 to300 seconds; the60-minute condition timeout remains unchanged.

No model, precision, noise, scene, prompt, order-within-arm, diffusion, cache or
success criterion changes. Earlier sources and all73 completed episodes remain
immutable. New launcher/helper hashes are recorded separately. On success the
combined completion marker supersedes the archived failed marker explicitly.
This resource exception is disclosed; latency may differ with light graphics
coexistence. No overall improvement claim precedes full paired analysis.
