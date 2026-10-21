# ex8-f_renderbinding_1021 — native renderer allocation probe

Preregistered before execution. This is a zero-scored-episode device-isolation experiment supporting ex8-d, not another full-suite attempt.

Question: does holding an explicit RenderSystem(device="cuda:0") after unchanged official CUDA/Curobo prewarm cause a subsequent unchanged native default scene to render on its assigned GPU, without any sampled peer GPU contexts?

Baseline: ex8-d frozen native runner SHA256 4b9094a29bb60011d77aa6ee0ed1cdcaa3edc6bba80a8cc63347c9f8073a8b3c; preserved failed blocks_ranking_rgb attempt, plus live adjust_bottle recovery and the existing CPU preparation at commit4f835241. Default SapienRenderer ignores its keyword arguments. This baseline establishes a resource interruption, not a policy score.

Hypothesis: explicit RenderSystem priming selects the assigned GPU for default native scenes. Competing explanation: Vulkan enumeration opens peer contexts or default scenes ignore priming. The probe measures both explicit/default-scene PCI identity and observed process contexts across all eight GPUs. A selected-device log alone cannot pass. No unprimed GPU rerun is budgeted; preserved actual baseline is retained. This is a mechanism test, not a causal performance comparison.

Intervention: one explicit RenderSystem retained after the native prewarm. Hold fixed official source782e3133f674449e2986754cfc5f7c4e204835a2, installed SAPIEN, GPU admission rules, default scene constructor and prewarm. A CUDA tensor additionally fails closed if the upstream prewarm swallowed a failure. No policy, scenes, planner, inference optimization, score or task seed changes. The candidate is not injected into existing workers.

Metric and prediction: explicit/default-scene PCI matches the assigned UUID, CUDA ordinal0, >=3 complete all-GPU samples, >=1 observed assigned context, zero observed peer contexts and no monitor error. Prediction: PCI matches; peer isolation remains uncertain. Failure or missing evidence leaves isolation unvalidated. Sampling cannot exclude activity between observations. Hardware torch2.7.0+cu128/Blackwell deviation remains disclosed.

Stages: L0 exact source and budget checks; L1 CPU adversarial gates; L1 GPU renderer-only probe; later separately registered L2/L3 native episodes required before L4 recovery. No scientific conclusion or recovery release follows solely from this probe. No task episodes or videos are produced by this renderer-only test.

Budget: dispatcher180 occupied seconds including admission and cleanup; probe alarm90 seconds to finish before the unchanged worker135-second guard. Existing conditional recovery reserve1007796 seconds plus180 seconds totals1007976 seconds (279.993333 GPUh), leaving24 seconds under the separate ex8-d280GPUh planning ceiling. Existing attempts retain every cap. Deadline2026-10-28T18:20:27.005560+00:00; no retries. Recompute reserve before freeze and submission. No additional headroom is implied by lower actual use until measured.

Stop: any pause, changed hash, budget/deadline failure, active native peer claim, unexpected own GPU process, contaminated evaluator environment, wrong PCI/device, sample error or timeout. Never stop another worker. Freeze a separate runtime and local registry from committed code; submit only through existing dispatcher after no-overlap precheck and idle30min. Keep the prepared job outside its queue while a native peer is active. Existing controller and queue remain unchanged.

Outputs: frozen protocol, exact source hashes, dispatcher-compatible prepared job, CPU validation and eventual result.json containing device records, all-context samples, errors and actual elapsed time. Full endpoint remains50 tasks x100 accepted episodes x2 modes; all prior failures and partial episodes remain preserved and excluded as registered.
