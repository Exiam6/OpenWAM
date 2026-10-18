# VAE deployment preflight failure and precision diagnosis

The registered VAE preflight stopped at 03:03:59 CDT on 18 October 2026. Cache construction and the three-step training smoke passed; deployment image/cache action equivalence failed its original 1e-5 threshold. The original failure remains unchanged. No tiny-fit stage, 20000-step training run, or closed-loop VAE evaluation was released.

The independent cache audit reloaded all 6424 BF16 latent windows, verified finite values and the cache digest, and matched actions, proprioception, text, task IDs and metadata exactly to the predecessor.

A separately registered, read-only policy diagnostic on the first adjust-bottle training window found image/cache action max error 0.0078125 and repeated-image error 0. Text, proprioception, schedules, initial action noise and future video noise matched. The actual image-path first-frame latent differed by 0.03515625 from the cached latent, although the standalone native-first-frame check outside autocast matched exactly.

An encoder-only controlled diagnostic then changed only the outer autocast context. Original encoding under autocast reproduced the discrepancy. The isolated patch disables autocast around the frozen encoder call while preserving its explicit dtype. All three prespecified training windows then matched the stored cache exactly:

| Training task | Original outer-autocast max error | Isolated fix max error |
|---|---:|---:|
| adjust_bottle | 0.03515625 | 0 (exact) |
| handover_block | 0.0390625 | 0 (exact) |
| place_object_basket | 0.046875 | 0 (exact) |

The candidate fix is committed as 00319e8c in the separate openwam-vae-control worktree. Two CPU regression tests verify context isolation, dtype preservation, no gradients and restoration of the caller context. The active openwam-rae-research engine and all registered GPU scripts remain unchanged. This establishes the encoder precision cause on the inspected windows; a corrected full-policy validation remains necessary before any new training decision.

The two diagnostic plans were recorded separately and never wrote a scientific release. Their raw arrays, captured forward inputs, logs and process records are retained. One raw encoder result has an escaped newline suffix; its bytes are preserved and the independent analysis validates and decodes the single JSON object.

Recorded occupied cost: 630.670 seconds (0.17519 GPU-hours), including the original failed preflight and both diagnostic parents.

Next work: preserve this failed study, validate any corrected protocol independently, and recheck full endpoint feasibility before considering three fresh training seeds. GPUs2/3/4 are now running the remaining original factorial jobs. Their hard caps must not be shortened or displaced to make a new endpoint appear feasible. Do not relax the deployment threshold or silently substitute diagnostic evidence for the failed gate.
