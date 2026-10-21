# Native Instance / physical / logical boundary preparation

Preparation only: no scientific attempt name, registration, GPU dispatch or endpoint score.

Question: if ex8-i first observes a peer context during native Context::Create, does a future independently registered diagnostic first observe it during Instance::Get, physical-device selection, or logical-device creation?

Baseline: ex8-i frozen at 0a56ba2429e07f7beb002e617dc4fce878cc55ec; its pending result must be read before selecting a new diagnostic. ex8-h preserved native prewarm success and three assigned-only samples after SAPIEN Device; peer first sampled at RenderSystem. Alternative explanations are Vulkan instance/enumeration/loader initialization versus native logical-device/allocator initialization.

Intervention candidate: expose the same installed native Instance::Get(app=0.0.1,engine=0.0.1,API=1.2), createPhysicalDevice("cuda:0"), createDevice() in separate monitored phases. The helper exposes the native selected PCI/CUDA ordinal and ray-tracing support for independent allocation verification, beyond the SAPIEN Device alias. Stop there: calling Context::Create afterward would duplicate the logical device and confound attribution. No native package, renderer defaults, ray tracing, planner, inference, scene or checkpoint changes. No library is loaded during CPU preparation.

A later GPU plan must retain explicit native CUDA/Curobo prewarm; verified private extension cache; assigned UUID and PCI checks before/after; complete all-card own-cgroup sampling and phase start/end timestamps; at least three complete samples after each boundary; immediate own-process stop on peer observation; pause/deadline/hash checks. No peer absence can be claimed between samples. Header/source compatibility and successful linking do not prove the exact installed binary source revision.

Prerequisites: ex8-i scientific, worker and claim terminal evidence must agree; official sync and compatibility must be renewed; isolated worktree plus committed preregistration and frozen hashes are required. Choose a unique scientific identity, never reset a claim. Recompute actual rounded terminal occupancy plus all original 97 held caps, conditional 18,749s blocks recovery and 9,000s native validation reserve under 280 GPUh and the unchanged 2026-10-28T18:20:27.005560+00:00 deadline. No GPU budget is allocated by this draft.

Possible outcomes: peer at Instance narrows to loader/enumeration; peer only at logical Device narrows to logical creation/allocator; assigned-only through all three leaves descriptor-pool/compiler or sequence effects unresolved. Each is a sampled interval finding, not exact source causality. All 97 full rows remain held. Native L2/L3 scene validation is mandatory before release; diagnostics never count toward the 10,000 episode endpoint.

If ex8-i passes, this candidate is unnecessary: preregister native L2/L3 isolation validation using the existing 9,000s reserve. If it times out or prewarm fails, diagnose that actual failure rather than treating Context as implicated.
