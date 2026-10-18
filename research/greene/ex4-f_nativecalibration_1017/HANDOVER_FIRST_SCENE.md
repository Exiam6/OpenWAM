# Tracking improves, but the first handover scene still receives no closing command

On scene6201000, all six fixed-endpoint policies fail under both the old and restored native executor. Restored calibration reduces median target-to-next-observation position error for each arm of each policy. Nevertheless, none of the corrected trajectories commands either gripper below0.5 during its800 action steps.

This is a post-hoc description of one shared scene across two models and three training seeds, not a separate hypothesis test. It identifies a concrete remaining behavior after the execution fix; it does not prove why the policy chooses that behavior. Other scenes may issue closing commands. The full pilot verdict remains pending in this snapshot.

Next: finish all fixed pilot scenes and the registered architecture comparison. Keep policy-command diagnostics separate from expert replay success and from task-level success rates.

![Native camera frames, commands and tracking errors](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/handover-first-scene/control-observation.png)

| Checkpoint | Left error: old → fixed (mm) | Right error: old → fixed (mm) | Corrected closing commands (left/right) |
|---|---:|---:|---:|
| raw768_native-seed42 | 11.50 → 2.15 | 1.18 → 0.54 | 0/0 |
| raw768_native-seed43 | 12.34 → 2.12 | 1.16 → 0.70 | 0/0 |
| raw768_native-seed44 | 8.35 → 1.80 | 1.20 → 0.49 | 0/0 |
| svae48_native-seed42 | 6.67 → 1.38 | 1.29 → 0.75 | 0/0 |
| svae48_native-seed43 | 9.98 → 1.61 | 1.26 → 0.72 | 0/0 |
| svae48_native-seed44 | 9.21 → 1.38 | 1.24 → 0.64 | 0/0 |

The tracking comparison uses the six newly rerun old-path controls. The videos below pair corrected episodes with their original historical old-path episodes; these are explicitly different old recordings. All controls use the same checkpoint, initial scene, native inference settings and task success rule. After different actions execute, trajectories diverge; the median tracking differences are descriptive, not matched-state causal estimates.

## All six paired videos

Old above, corrected below; head, left wrist and right wrist views. Action-index alignment at12fps; shorter recordings hold their last frame. Every clip fully decodes. The complete figure and two seed43 contact sheets were visually inspected at sampled start/intermediate/final frames. Native cameras do not guarantee a complete external view of the robot, and wrist images may be occluded.

| Checkpoint | Video |
|---|---|
| raw768_native-seed42 | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/handover_block/seed-6201000.mp4) |
| raw768_native-seed43 | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/handover_block/seed-6201000.mp4) |
| raw768_native-seed44 | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/handover_block/seed-6201000.mp4) |
| svae48_native-seed42 | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/handover_block/seed-6201000.mp4) |
| svae48_native-seed43 | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/handover_block/seed-6201000.mp4) |
| svae48_native-seed44 | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/handover_block/seed-6201000.mp4) |

[Full-quality evidence bundle](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/handover-first-scene-evidence.zip)
