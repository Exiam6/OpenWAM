# ex8-h_deviceboundary_1021 — split native device discovery from render construction

Question: after successful native prewarm, is the first sampled peer context already present after explicit Device(cuda:0), before RenderSystem construction?

Baseline: ex8-g frozen commit923bc560 and preserved scientific/worker failures. Prewarm succeeded; explicit RenderSystem selected assigned PCI, but its phase showed peer graphics8MiB. ex8-d original full runner and corrected classifier remain frozen;97 rows stay held,23 interrupted episodes stay excluded.

Hypothesis: device discovery creates the peer context before a render system is constructed. Competing explanation: rendering-system construction creates it. Intervention: replace RenderSystem(device="cuda:0") with Device("cuda:0"), a monitored checkpoint, then RenderSystem(device=device). Installed stubs explicitly expose both overloads. No environment filtering, package changes or rendering semantics changes. This is a diagnostic, not a production isolation repair.

Held fixed: official782e3133 (real pull unchanged), native prewarm and exact ex8-d build-tool environment, separate cache, installed SAPIEN3.0.0b1 and exact native library hashes, UUID assignment, PCI verification, all-card ownership monitor, original Engine/renderer/scene sequence. No tasks, inference, scene tuning, planner fallback or episodes.

Metric: earliest phase with sampled owned peer context, all complete samples, Device and RenderSystem PCI. Prediction: discovery may be the boundary; unknown until measured. >=3 samples per completed boundary. Stop immediately on owned peer context or monitor error; preserve failure and phase. Sampling localizes observed boundaries only and cannot prove exact internal causality or continuous absence. No-peer outcome requires all boundaries and successful worker result; still cannot release held full jobs or establish L2/L3.

L0: source diff changes only split constructor and new identity/accounting; exact file hashes and native API checked. L1 CPU: prewarm failure tests, admission, PCI mismatch and budget checks, boundary order. L1 GPU pending. L2/L3 native scene validation requires separate later preregistration and successful isolation intervention. L4 remains blocked.

Budget:300 occupied seconds including all startup/build/monitor/cleanup;210s child alarm,230s launcher within255s worker guard. Seven-day original deadline unchanged. Before registration sum all rounded terminal d/e/f/g/h occupancy +97 held caps +18749s conditional blocks recovery +9000s conditional native validation +300s this diagnostic, <=280GPUh. Preserve original caps; no retries or claim replacement. This leaves only64s at current ledger, so future stages must be reconciled against actual occupancy before registration.

Stop on global/job pauses, changed source/hash, stale inventory, active/uncertain native peer claim, prewarm failure, peer context, bad PCI, incomplete samples, deadline or budget excess. Launch only via immutable existing dispatcher, idle30min and current window rules. Local committed registry/protocol/prepared job only; no external upload. All previous attempts and failures retained. Three states are separate: prepared, terminal worker, validated conclusion.

Source provenance limit: package metadata says3.0.0b1 while generated stub embeds3.0.0.dev20240429+7809345c. That revision is a source candidate, not proven binary provenance. Store exact native hashes and candidate header comparisons; do not treat matching headers alone as proof of exact binary build revision.
