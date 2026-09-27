# Authorized recovery with current idle GPU placement

User instruction: `continue`, following the concrete three-job recovery approval request. The previous proposal is preserved at `preflight-recovery-20260924T0315.json`.

Fresh resource checks show GPU4 is now in use by another user. The same three zero-training seed44 tasks will run once on cm009 L40S GPU3 (S-VAE), GPU5 (PCA), GPU6 (Wan), subject to three clean idle checks. Only GPU index/UUID metadata changes; seeds, training/source/configuration,150scenes,5400rollouts, success criteria and original deadlines remain fixed. All original startup failures and controller logs are retained in a new attempt archive. No automatic retry after this recovery.

This adds the explicitly authorized operational exception to frozen no_restart_or_replay without modifying the original plan or scientific protocol.
