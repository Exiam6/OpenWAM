# From representation diagnosis to control improvement

2026-09-21. Proposed extension; no training or closed-loop improvement established.

## Evidence and unresolved questions

The completed study includes a proprioception-only reference. Adding compressed visual features reduces the clean held-out state-change probe's normalized MSE in all three tasks. The numbers below are a re-analysis of existing results, not additional runs. They support the usefulness of vision for this particular readout; they do not isolate pretraining causally, prove the bottleneck loses information, or measure policy success.

[
  {
    "task": "adjust_bottle",
    "proprio_only_mse": 0.2416046111620834,
    "vision_plus_proprio_mse": 0.13593767985429808,
    "relative_error_reduction_percent": 43.73547789487237
  },
  {
    "task": "handover_block",
    "proprio_only_mse": 0.3999962754726339,
    "vision_plus_proprio_mse": 0.10950001638662617,
    "relative_error_reduction_percent": 72.62474100359024
  },
  {
    "task": "place_object_basket",
    "proprio_only_mse": 0.465277444189409,
    "vision_plus_proprio_mse": 0.22125745761509652,
    "relative_error_reduction_percent": 52.44612426880827
  }
]

The existing experiment compares PCA-48 and S-VAE-48, not raw DINO versus compressed DINO. Therefore it does not yet locate information loss in the compressor. Raw-feature probes should use train-only normalization and validation-selected regularization; different input dimensions change probe capacity, so treat that comparison as a diagnostic, not a capacity-matched causal estimate.

## Candidate C: control-relevant supervision during compression

A conditional next hypothesis: if compressed features lose readable task information available upstream, add a small auxiliary predictor to S-VAE training. Freeze DINO; train the compressor and auxiliary head jointly. Initially preserve the 48-channel bottleneck, original reconstruction/cosine/KL terms, data split and budget.

    z_t = compress(DINO(current_image))
    prediction = auxiliary_head(z_t, current_proprioception)
    loss = original_SVAE_loss + lambda * normalized_state_change_loss

Use the existing label first: achieved end-effector translation change over four raw steps and achieved gripper state at that horizon. These are trajectory state-change targets, not expert control commands. Future observations/latents must not be inputs to the auxiliary head; only the current latent and current state are allowed. Task instruction may be included only if available and equally provided to comparison methods. Future states appear only as training labels. Use deterministic current posterior mean for the first ablation and document this choice.

The auxiliary head is only for training, so no additional head is required at inference. However, changing the compressor still changes the latent coordinates. A trained controller is not automatically compatible just because shape and inference head count stay fixed. Expect to validate latent alignment and/or afford limited policy adaptation. Do not claim drop-in improvement.

Start with one auxiliary loss and no new layer fusion or denoising. Predeclare lambda candidates 0, 0.1, 1 on standardized target dimensions; validation selects the candidate and test remains untouched. Lambda=0 retains the same auxiliary-head plumbing while detaching it from the compressor objective, with separate random streams and identical compressor minibatches/initialization so it numerically reproduces the existing baseline. The original published-policy baseline remains a separate inference evaluation; a fresh per-task compressor is not that published checkpoint.

Validation has only five episodes per task in the existing split. Limit the search, preserve the final checkpoint rule, report all seeds, and treat the chosen coefficient as exploratory until replicated on fresh episodes. The old held-out episodes have already informed method choice, so repeated use is a development benchmark rather than a fully untouched confirmation set. Obtain fresh episodes or a preregistered unseen split for confirmatory claims.

## Controls that prevent misleading wins

- Compare the original compressor with the auxiliary-trained compressor at equal feature width and training budget. Keep layer fusion and denoising out of this comparison.
- Report proprioception-only, aligned visual plus proprioception, and held-out visual-shuffling diagnostics. Shuffle within each task while retaining the state and label; document that this intervention creates mismatched pairs and is a dependence diagnostic, not a deployment condition.
- Do not report only the jointly trained head's accuracy. Freeze each encoder and refit an independent, identically specified probe using training episodes; select regularization on validation and evaluate on held-out episodes.
- Check whether additional targets or new tasks benefit. The supervised target improving does not establish that all task-relevant information is preserved.
- Report clean and perturbed readouts, full feature reconstruction and event coverage, including regressions. Never call gripper transitions actual contact events without validation.
- A successful result ultimately requires paired closed-loop rollouts after explicitly resolving policy compatibility. Compare equivalent controller adaptation budgets when retraining or fine-tuning is needed; retain an unchanged-policy reference.

## Choose the intervention based on the failure

| Diagnostic pattern | Candidate | Intended mechanism |
| --- | --- | --- |
| Multi-layer DINO exposes task information less readable from the last layer | A: multi-layer features | Improve information available before compression |
| Raw features support stronger task readouts than the bottleneck, after careful controls | C: control-supervised compression | Allocate the limited latent capacity to task information |
| Clean features work but sensor noise displaces them | B: original-coordinate observation denoising | Stabilize input for the existing policy |
| Both raw and compressed readouts are weak | Inspect temporal context, labels, instruction and observability | Avoid expecting a compressor to invent missing information |

Real viewpoint changes require preserving camera-dependent geometry or using valid correspondence/calibration. Do not impose pixel-position invariance across cameras. A small local adapter may be useful for noise without solving viewpoint transfer.

## Priority

Complete the published-policy perturbation baseline and the missing raw-feature diagnostic first. Candidate C directly tests the mismatch between reconstruction and the control readout; Candidate B remains the cheaper route to testing robustness with an unchanged policy. Candidate A addresses the upstream feature source. Choose one on diagnostic evidence, not by stacking all three. These are proposed experiments, not claimed novel methods or measured improvements.
