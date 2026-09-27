# Reproduce the frozen normalization follow-up

Use the same runtime, model and simulator assets as REPRODUCE_ROBUSTNESS.md.
Read NORM_PROTOCOL.md first. All four arms reuse the original selected adapter
and its fixed hash. This round has no training, no new model selection and no
held-out demonstration extraction.

1. Put the new norm scripts beside robustness_common.py/robustness_client.py in
   the runtime scripts directory. Keep prior experiment outputs immutable.
2. Use a new results directory for repetition. Preserve the frozen settings;
   adapt hard-coded paths/GPU UUID only for your machine and document changes.
3. Run check_norm_intervention.py and check_norm_replay.py in policy-env.
   These check the actual encoder normalization method and execute the original
   transformed RoboTwin loop using a fake environment. No simulator success
   inference can be drawn from these CPU checks.
4. Run norm_offline.py for the existing five-episode validation-cache diagnostic.
   The cache corruption is sigma=.10; closed-loop stress in this round is .20.
5. Run run-norm-study.sh. It guards one GPU with idle/ECC/process checks and flock,
   starts one server per arm, runs 10 original-policy feasibility references,
   freezes the first 10 accepted scene seeds/prompts independently of success,
   then replays all four arms on 10 clean + 10 noisy scenes. It also runs the
   single selected prior failing scene separately for every arm: 94 total,
   80 primary. Each condition has a 60-minute timeout and logs every outcome.
6. Require all launcher exit codes to be zero. Run summarize_norm.py; it checks
   frozen source and weight hashes, simulator counts, prompt/image/state pairing,
   fixed manifests and all 94 videos. Native RoboTwin result files still divide
   capped counts by 100; summaries use the actual number of completed episodes.
7. Run build_norm_report.py from the research repository. NORM_SITE_DIR can point
   to an isolated copy of the static Site when making a local final report;
   publication is a separate action. Partial reports are labeled snapshots.

On another renderer, establish a fresh reference rather than silently bypassing
image/state mismatches. Historical seed300003 is deliberately selected for a
known prior failure: never pool it into fresh-scene success rates. Fresh prompt
selection consumes a 10-item NumPy choice; the historical scene preserves its
prior 5-item choice even when run as a single episode.

Interpretation: renorm vs identity controls repeated normalization's rounding
and numerical effects. Adapter_renorm vs adapter is the primary comparison for
clean and noise separately. A rescue of one selected failure alone is not a
benefit claim. Neither this experiment nor the prior residual study establishes
RAE-vs-VAE superiority or contact robustness.
