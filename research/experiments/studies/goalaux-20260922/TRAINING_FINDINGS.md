# Six matched fits complete; confirmatory benefit unmeasured

Recorded 2026-09-22T16:38:30.236754-05:00.
Training exit0; independent artifact audit exit0/passed. Each pair starts from
the same L12 checkpoint and shares its complete sample sequence and input stats.
All final tensors are finite. Encoder weights differ between arms; this confirms
that training affected the representation, not that it improved a useful metric.

| Arm | Seed | Updates | Fit wall seconds | Peak allocated GPU GB |
|---|---:|---:|---:|---:|
| reconstruction_only | 42 | 2000 | 45.6 | 6.90 |
| goal_auxiliary | 42 | 2000 | 60.5 | 6.99 |
| reconstruction_only | 43 | 2000 | 46.5 | 6.90 |
| goal_auxiliary | 43 | 2000 | 60.7 | 6.99 |
| reconstruction_only | 44 | 2000 | 44.8 | 6.90 |
| goal_auxiliary | 44 | 2000 | 60.8 | 6.99 |

Data loading occurred before these per-fit times. The full loaded training cache
raises peak memory above the small three-episode GPU timing preflight. The final
GPU snapshot is1MiB,0%utilization,0uncorrectedECC: this job released its device.

105original training episodes and1260clips were used; original development/fresh
scenes were not used in these fits. The auxiliary head matches the existing
readout's per-token normalization and is removed at inference. Heads/readouts
for evaluating the newly fitted encoders have NOT yet been fitted. No fresh
confirmation scene from the800000/801000/802000ranges has been generated/scored
by this study. No held-out benefit, state preservation or policy benefit is known.

Next: implement the already fixed fresh-collection/readout protocol, verify scene
seed exclusion and native generator equivalence, and freeze readout/checkpoint
hashes before any new confirmation scores. Collect only after idle-resource and
source checks, within the original01:15:12CDT Sep23deadline. Do not repeat these
six fits, completed audits, old80rollouts, or the stopped context pilot.
