# User steering: a small representation contribution that improves policy success

Recorded: 2026-09-26T16:51:43.496866-05:00
User: “openwam在乎的指标是什么 我关键是要提点”.

The desired deliverable is a compact change with verified gains in OpenWAM policy success. Software diagnostics, reconstruction losses and offline state/goal readouts support diagnosis but do not satisfy that objective by themselves. No such policy improvement has been established by our experiments yet. This document records priority and a candidate, not a launched or frozen scientific protocol.

## Primary-source grounding

OpenWAM v1, https://arxiv.org/pdf/2609.07398 :
- p11, study evaluation: RoboTwin2.0-Full and Clean2Random, clean and randomized success rates. Full trains with2500 clean+25000 randomized demonstrations; Clean2Random trains on clean only. Our3-task custom-noise/yaw study is neither complete official protocol.
- pp12–13, representation study: frozen representation encoders with an independently initialized Wan5B architecture; temporal compression by averaging groups of4frames; S-VAE contracts features to48channels. Compressed DINO is nearly on par with Wan VAE. The paper explicitly encourages representation encoders with native temporal compression.
- p22, Table5: OpenWAM-alpha LIBERO-Plus camera33.8%, noise39.8%, average69.2%. These numbers concern the released full-scale policy and official benchmark; they are not baselines directly comparable to our reduced295M3-task study.
- p33, limitation4: compact latents retaining sufficient environmental information and robustness to viewpoint/noise/scene changes remain open research questions.

## What would count as improvement

Primary: paired held-out task success-rate difference against the native DINO+S-VAE baseline under an explicitly declared training/evaluation setting. Retain clean performance and report perturbation-specific and aggregate success, all tasks, all seeds and uncertainty. Latency, memory and trainable parameters are cost constraints. Report gains in percentage points:70% to75% means+5pp. Do not treat offline error reduction as success-rate improvement.

A small subset can justify only a subset-scoped claim. An official benchmark claim requires its original protocol and coverage. A reduced-model result is not proof of OpenWAM-alpha/5B improvement. To claim only a modest code contribution with useful performance, a controlled representation-branch improvement can be sufficient; it must name that scope and baseline honestly.

## Prioritized next actions

1. Finish or truthfully close the existing frozen3-route/3-seed native295M evaluation. Do not add further offline probe sweeps while policy success is unknown. First establish that the matched baselines learn usable policies; a floor across all routes cannot rank representations. Existing resource assignments, deadlines, stop rules and completed data remain unchanged.
2. If a new intervention is needed, prioritize ONE candidate: compact causal temporal compression replacing fixed4-frame mean pooling in the existing DINO+S-VAE route. Keep the first conditioning frame isolated, latent channel width48, token count, action interface and inference schedule fixed. Initialize any learnable aggregation at the current mean-pooling behavior. The hypothesis is that averaging suppresses rapid motion or gripper-transition information; this mechanism is not yet established. Determine a non-collapsing training objective on training data before freezing a bounded comparison. Same-step/budget baseline and matched updates of downstream models are required; no evaluation-time encoder swap into a mismatched pretrained policy.
3. Any new candidate selection/hyperparameter work uses development data only. Use a prospectively fixed new test cohort for confirmation; do not tune on the existing held-out matrix. Benchmark-scale adaptation follows only if the small controlled test supports it. No guaranteed gain or promised+Xpoints.

Do not train a pixel decoder solely to improve unchanged frozen control features: decoder-only reconstruction training does not change the features supplied to the policy. Current PCA remains a useful baseline, not a proven winning candidate. Earlier failed/incomplete auxiliary-loss experiments remain retained and cannot be presented as validated gains.

This note creates no new job, timer, agent, deadline extension, altered frozen protocol, upstream PR or maintainer message.
