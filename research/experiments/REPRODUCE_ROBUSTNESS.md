# Reproduce the published-policy robustness pilot

Use the fixed assets and environments documented in POLICY_SMOKE_PROTOCOL.md
and the policy/benchmark requirement locks. The scripts assume an experiment
root with `OpenWAM`, `benchmarks/RoboTwin`, `assets/dinov3-policy`,
`assets/dinov3-study/encoder.safetensors`, and the 50 pick_dual_bottles demos.
Copy scripts into that root; use a new output directory/date for repetitions.
Do not overwrite previous outputs. Adjust resource UUIDs and paths for your
machine, preserving all model, intervention and seed settings.

1. Read ROBUSTNESS_PROTOCOL.md, including the instruction-RNG correction.
2. Start run-robustness.sh baseline and run-restoration.sh on separate idle GPUs.
   The standalone encoder waits for the full-policy numerical reference.
3. Use run-fixed-confirmation.sh for the final protocol: verify the accepted-scene
   and prompt manifest in fixed-scenes.json and all initial image/state hashes.
   It runs all four arms with the official action loop, disabling only repeated
   expert re-filtering after feasibility was established in the reference.
   The initial serial confirmation was stopped because seed 300002 was accepted
   in clean but skipped in noise before corruption; all original records remain
   under invalid-refiltering-attempt. The earlier GPU0 scheduling attempt also
   failed preflight before any rollout. Do not run legacy launchers alongside
   the final fixed-scene launcher. See fixed-scene-amendment.md in the report.
4. Run scripts/summarize_robustness.py from the runtime root's policy environment.
   It refuses to summarize incomplete or unmatched pairs and checks source,
   selected-weight and selection hashes; it uses real episode counts, retaining
   the upstream capped-evaluation result-file denominator error as evidence.
5. Build the public report with build_robustness_report.py from this repository.

The first clean-only attempt was invalid for paired analysis because upstream
language generation was not seeded. All its artifacts are kept under
invalid-unpaired-attempt; they are excluded from the 35 paired rollouts. A partial
next-condition process was stopped before its comparisons were used. This was an
operational correction, not model selection or selective outcome exclusion.

Adapter training: two architectures x three seeds, 2,000 steps each. Zero-init
linear/MLP residuals only change the observed latent frame. Original encoder,
compressor, normalization and policy remain frozen. Targets are published clean
coordinates. HDF5 test episodes are never opened for this fit. The closed-loop
confirmation uses separate simulator seeds from the baseline matrix.

Statistics: report exact paired discordances and two-sided binomial p values;
Wilson intervals are descriptive per-condition binomial intervals. At n=5,
uncertainty is large. There is no multi-task or physical-robot success claim.

The archived manifest includes this run's exact rendered-image/proprio hashes.
On a different renderer/driver stack, establish a new clean reference before
any model comparison, preserve the same scene seeds and prompts, and archive
that reference. Do not silently bypass image/state mismatches.

Operational note: the immediate identity-to-adapter handoff saw 6 MiB memory but
a stale 22% utilization sample and aborted before loading the adapter server.
After verifying 0% utilization/no GPU processes, only the untouched adapter phase
was relaunched with the same guard and completed. No episodes were dropped.
