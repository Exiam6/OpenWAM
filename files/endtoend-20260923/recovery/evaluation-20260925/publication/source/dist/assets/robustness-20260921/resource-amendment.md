2026-09-21: after extraction/training completed, reuse GPU0 for all four frozen
confirmation conditions while the baseline camera-transfer condition finishes
on GPU2. Two GPUs maximum is unchanged. All within-confirmation pairs remain on
GPU0 with the same seeds, checkpoint, inference settings and selected adapter;
this is a resource/scheduling correction only. Confirmation's localhost port is
18849; baseline uses 18848. No model selection depends on baseline outcomes.

Correction at 13:52 CDT: GPU0 was claimed by other processes before the launch
preflight. The preflight saw 8,709 MiB used and stopped before constructing a
policy server or running any confirmation episode. Return to the original
serial launcher on GPU2 after the baseline. No other user's process was changed;
no confirmation outcomes were observed or excluded. The failed GPU0 launcher log
and initial unlock record remain saved. The final confirmation runs use GPU2.
