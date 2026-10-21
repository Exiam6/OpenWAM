# Native render-device binding preparation

CPU preparation only, registered 2026-10-21 before validation. No scientific row, GPU claim, dispatch or runtime modification is authorized by this file alone.

Question: can a standalone launch adapter request the assigned SAPIEN render device after native CUDA/Curobo prewarming, then delegate to the unchanged native evaluator without selecting Labtasker, changing scenes, planner, inference, accepted-seed selection or scoring?

Baseline: frozen ex8-d runner 4b9094a29bb60011d77aa6ee0ed1cdcaa3edc6bba80a8cc63347c9f8073a8b3c; native wrapper in ex8-a. Original failed blocks_ranking_rgb row and worker results remain evidence. Official sync 2026-10-21T20:16:04Z unchanged at782e3133f674449e2986754cfc5f7c4e204835a2, compatible with frozen sources; no experiment source upgraded.

Hypothesis: priming RenderSystem(device="cuda:0") after prewarm selects the allocation's PCI device for subsequent default native scenes. Competing explanation: Vulkan device enumeration still opens peer graphics contexts, or default scenes ignore priming. CPU tests can verify ordering/delegation/fail-closed checks only; only a later GPU probe can distinguish the hypotheses.

Intervention candidate: hold one explicit RenderSystem while invoking the exact original native main. Hook only the wrapper's prewarm function; do not patch scene constructors, task code, policy, inference, planner, error classifiers or scoring. Legacy sapien.wrapper.renderer.SapienRenderer ignores kwargs; adding device= there is not a repair. Device PCI and CUDA mapping must match the assigned UUID. Preserve version check and reject all Labtasker, episode-count and planner-fallback environment settings. Record selected device and later default native scene device separately in the probe.

L0: hash and inspect baseline APIs; verify frozen native dispatch remains untouched. L1 CPU: adversarial stub tests for import/order, native argv, missing/ambiguous UUID, contaminated env, wrong PCI/CUDA id and exception propagation. L2/L3 GPU: NOT RUN, NOT PASSED. A separate frozen dispatcher-only probe must establish actual default-scene device and all-process GPU contexts, including transient peer contexts. A matching selected-device log alone cannot pass isolation. No full-row recovery before GPU isolation evidence and explicit frozen endpoint selection.

Budget now <=120 CPU seconds, zero GPU hours, deadline this takeover. Proposed future probe <=180 occupied seconds, no policy scoring; together with the existing conditional recovery reserve1007796 seconds this is1007976 seconds (279.993333 GPUh), leaving24 seconds under280GPUh. No existing cap changes. Deadline stays2026-10-28T18:20:27.005560+00:00. Any changed accounting requires a fresh scoped assessment, never silently enlarging this stage.

Stop on any global/registered pause, unexpected environment, API mismatch, wrong device or exception. Never inject this into a running process or edit frozen runtimes/controller/jobs. Current legitimate recovery continues. Outputs: committed adapter/probe/tests, immutable source hashes, CPU validation and actionable deferred probe gates. Prediction: CPU checks pass; actual isolation remains undecided until the probe. This preparation cannot establish official-score reproduction or representation superiority.
