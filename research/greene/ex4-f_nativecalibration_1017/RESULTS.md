# Native-calibration learned-policy pilot

Completed:90 corrected episodes and six reproducing old-path controls. Registered primary verdict: **no_handover_success_on_pilot**.

The question is whether restoring verified native planner calibration raises learned-policy handover success above zero. The intervention preserves the ALOHA filename marker when relocating numerically identical planner configurations. Checkpoints at step20000, native inference, prompts, RNG, first-five frozen scenes per task, cameras and task success rules are unchanged. This is a development pilot with three training seeds and five shared scene IDs per task, not 90 independent scenes or an unseen test.

The preregistered primary criterion is at least two handover successes out of15 for either model. Zero for both leaves unresolved learned-policy behavior after the execution fix; one success is undecided. An adjust success drop of at least20 percentage points is flagged as regression. Place is exploratory. This pilot cannot rank RAE architectures or invalidate the separate successful expert-execution repair. It does not isolate training, model choice, input handling or deployment feedback as the cause of remaining failures.

Actual occupied GPU time: 4.302hours, within the registered6-hour cap. Seed42 cells and the four safely handed-off continuation cells are counted once; idle admission waits are excluded.

| Model | Task | Historical old execution | Restored calibration | Gains | Losses | Regression flag |
|---|---|---:|---:|---:|---:|---|
| raw768_native | adjust_bottle | 33% (5/15) | 100% (15/15) | 10 | 0 | False |
| raw768_native | handover_block | 0% (0/15) | 0% (0/15) | 0 | 0 | False |
| raw768_native | place_object_basket | 0% (0/15) | 0% (0/15) | 0 | 0 | False |
| svae48_native | adjust_bottle | 47% (7/15) | 100% (15/15) | 8 | 0 | False |
| svae48_native | handover_block | 0% (0/15) | 0% (0/15) | 0 | 0 | False |
| svae48_native | place_object_basket | 0% (0/15) | 0% (0/15) | 0 | 0 | False |

Next: follow the separately registered corrected factorial to compare the existing architectures under the same repaired executor. Keep execution recovery and learned-policy behavior as distinct claims. Do not change model selection, task limits or scene membership in response to these results.

## Every training seed

| Model | Seed | Task | Old | Fixed |
|---|---:|---|---:|---:|
| raw768_native | 42 | adjust_bottle | 40% (2/5) | 100% (5/5) |
| raw768_native | 42 | handover_block | 0% (0/5) | 0% (0/5) |
| raw768_native | 42 | place_object_basket | 0% (0/5) | 0% (0/5) |
| raw768_native | 43 | adjust_bottle | 40% (2/5) | 100% (5/5) |
| raw768_native | 43 | handover_block | 0% (0/5) | 0% (0/5) |
| raw768_native | 43 | place_object_basket | 0% (0/5) | 0% (0/5) |
| raw768_native | 44 | adjust_bottle | 20% (1/5) | 100% (5/5) |
| raw768_native | 44 | handover_block | 0% (0/5) | 0% (0/5) |
| raw768_native | 44 | place_object_basket | 0% (0/5) | 0% (0/5) |
| svae48_native | 42 | adjust_bottle | 40% (2/5) | 100% (5/5) |
| svae48_native | 42 | handover_block | 0% (0/5) | 0% (0/5) |
| svae48_native | 42 | place_object_basket | 0% (0/5) | 0% (0/5) |
| svae48_native | 43 | adjust_bottle | 60% (3/5) | 100% (5/5) |
| svae48_native | 43 | handover_block | 0% (0/5) | 0% (0/5) |
| svae48_native | 43 | place_object_basket | 0% (0/5) | 0% (0/5) |
| svae48_native | 44 | adjust_bottle | 40% (2/5) | 100% (5/5) |
| svae48_native | 44 | handover_block | 0% (0/5) | 0% (0/5) |
| svae48_native | 44 | place_object_basket | 0% (0/5) | 0% (0/5) |

## Validation and limits

All96 episode results pass independent auditing; the90 corrected pairs have exact initial states, zero initial camera error in all three views and matching first native action chunks. All90 videos fully decode and match result hashes. The first registered scene contact sheet of every model/seed/task group was visually reviewed at sampled start/intermediate/final frames. This is not manual inspection of every video frame. Native wrist cameras can be occluded; no additional external whole-robot view was recorded.

Videos pair historical old execution above and restored calibration below. Camera order is head, left wrist, right wrist. Clips align action index at12fps rather than physical time; shorter episodes hold their final frame. All failed outcomes and the earlier intermediate evidence bundles remain available. No repeated failed attempt is silently substituted.

## All90 paired clips

| Model and seed | Task | Scene | Old | Fixed | Video |
|---|---|---:|---|---|---|
| raw768_native-seed42 | adjust_bottle | 6200001 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200001.mp4) |
| raw768_native-seed42 | adjust_bottle | 6200002 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200002.mp4) |
| raw768_native-seed42 | adjust_bottle | 6200003 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200003.mp4) |
| raw768_native-seed42 | adjust_bottle | 6200006 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200006.mp4) |
| raw768_native-seed42 | adjust_bottle | 6200007 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200007.mp4) |
| svae48_native-seed42 | adjust_bottle | 6200001 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200001.mp4) |
| svae48_native-seed42 | adjust_bottle | 6200002 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200002.mp4) |
| svae48_native-seed42 | adjust_bottle | 6200003 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200003.mp4) |
| svae48_native-seed42 | adjust_bottle | 6200006 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200006.mp4) |
| svae48_native-seed42 | adjust_bottle | 6200007 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200007.mp4) |
| raw768_native-seed43 | adjust_bottle | 6200001 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200001.mp4) |
| raw768_native-seed43 | adjust_bottle | 6200002 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200002.mp4) |
| raw768_native-seed43 | adjust_bottle | 6200003 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200003.mp4) |
| raw768_native-seed43 | adjust_bottle | 6200006 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200006.mp4) |
| raw768_native-seed43 | adjust_bottle | 6200007 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200007.mp4) |
| svae48_native-seed43 | adjust_bottle | 6200001 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200001.mp4) |
| svae48_native-seed43 | adjust_bottle | 6200002 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200002.mp4) |
| svae48_native-seed43 | adjust_bottle | 6200003 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200003.mp4) |
| svae48_native-seed43 | adjust_bottle | 6200006 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200006.mp4) |
| svae48_native-seed43 | adjust_bottle | 6200007 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200007.mp4) |
| raw768_native-seed44 | adjust_bottle | 6200001 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200001.mp4) |
| raw768_native-seed44 | adjust_bottle | 6200002 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200002.mp4) |
| raw768_native-seed44 | adjust_bottle | 6200003 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200003.mp4) |
| raw768_native-seed44 | adjust_bottle | 6200006 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200006.mp4) |
| raw768_native-seed44 | adjust_bottle | 6200007 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200007.mp4) |
| svae48_native-seed44 | adjust_bottle | 6200001 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200001.mp4) |
| svae48_native-seed44 | adjust_bottle | 6200002 | True | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200002.mp4) |
| svae48_native-seed44 | adjust_bottle | 6200003 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200003.mp4) |
| svae48_native-seed44 | adjust_bottle | 6200006 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200006.mp4) |
| svae48_native-seed44 | adjust_bottle | 6200007 | False | True | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200007.mp4) |
| raw768_native-seed42 | handover_block | 6201000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/handover_block/seed-6201000.mp4) |
| raw768_native-seed42 | handover_block | 6201001 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/handover_block/seed-6201001.mp4) |
| raw768_native-seed42 | handover_block | 6201002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/handover_block/seed-6201002.mp4) |
| raw768_native-seed42 | handover_block | 6201003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/handover_block/seed-6201003.mp4) |
| raw768_native-seed42 | handover_block | 6201004 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/handover_block/seed-6201004.mp4) |
| svae48_native-seed42 | handover_block | 6201000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/handover_block/seed-6201000.mp4) |
| svae48_native-seed42 | handover_block | 6201001 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/handover_block/seed-6201001.mp4) |
| svae48_native-seed42 | handover_block | 6201002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/handover_block/seed-6201002.mp4) |
| svae48_native-seed42 | handover_block | 6201003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/handover_block/seed-6201003.mp4) |
| svae48_native-seed42 | handover_block | 6201004 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/handover_block/seed-6201004.mp4) |
| raw768_native-seed43 | handover_block | 6201000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/handover_block/seed-6201000.mp4) |
| raw768_native-seed43 | handover_block | 6201001 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/handover_block/seed-6201001.mp4) |
| raw768_native-seed43 | handover_block | 6201002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/handover_block/seed-6201002.mp4) |
| raw768_native-seed43 | handover_block | 6201003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/handover_block/seed-6201003.mp4) |
| raw768_native-seed43 | handover_block | 6201004 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/handover_block/seed-6201004.mp4) |
| svae48_native-seed43 | handover_block | 6201000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/handover_block/seed-6201000.mp4) |
| svae48_native-seed43 | handover_block | 6201001 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/handover_block/seed-6201001.mp4) |
| svae48_native-seed43 | handover_block | 6201002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/handover_block/seed-6201002.mp4) |
| svae48_native-seed43 | handover_block | 6201003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/handover_block/seed-6201003.mp4) |
| svae48_native-seed43 | handover_block | 6201004 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/handover_block/seed-6201004.mp4) |
| raw768_native-seed44 | handover_block | 6201000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/handover_block/seed-6201000.mp4) |
| raw768_native-seed44 | handover_block | 6201001 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/handover_block/seed-6201001.mp4) |
| raw768_native-seed44 | handover_block | 6201002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/handover_block/seed-6201002.mp4) |
| raw768_native-seed44 | handover_block | 6201003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/handover_block/seed-6201003.mp4) |
| raw768_native-seed44 | handover_block | 6201004 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/handover_block/seed-6201004.mp4) |
| svae48_native-seed44 | handover_block | 6201000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/handover_block/seed-6201000.mp4) |
| svae48_native-seed44 | handover_block | 6201001 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/handover_block/seed-6201001.mp4) |
| svae48_native-seed44 | handover_block | 6201002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/handover_block/seed-6201002.mp4) |
| svae48_native-seed44 | handover_block | 6201003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/handover_block/seed-6201003.mp4) |
| svae48_native-seed44 | handover_block | 6201004 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/handover_block/seed-6201004.mp4) |
| raw768_native-seed42 | place_object_basket | 6202000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/place_object_basket/seed-6202000.mp4) |
| raw768_native-seed42 | place_object_basket | 6202002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/place_object_basket/seed-6202002.mp4) |
| raw768_native-seed42 | place_object_basket | 6202003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/place_object_basket/seed-6202003.mp4) |
| raw768_native-seed42 | place_object_basket | 6202006 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/place_object_basket/seed-6202006.mp4) |
| raw768_native-seed42 | place_object_basket | 6202007 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/place_object_basket/seed-6202007.mp4) |
| svae48_native-seed42 | place_object_basket | 6202000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/place_object_basket/seed-6202000.mp4) |
| svae48_native-seed42 | place_object_basket | 6202002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/place_object_basket/seed-6202002.mp4) |
| svae48_native-seed42 | place_object_basket | 6202003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/place_object_basket/seed-6202003.mp4) |
| svae48_native-seed42 | place_object_basket | 6202006 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/place_object_basket/seed-6202006.mp4) |
| svae48_native-seed42 | place_object_basket | 6202007 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/place_object_basket/seed-6202007.mp4) |
| raw768_native-seed43 | place_object_basket | 6202000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/place_object_basket/seed-6202000.mp4) |
| raw768_native-seed43 | place_object_basket | 6202002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/place_object_basket/seed-6202002.mp4) |
| raw768_native-seed43 | place_object_basket | 6202003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/place_object_basket/seed-6202003.mp4) |
| raw768_native-seed43 | place_object_basket | 6202006 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/place_object_basket/seed-6202006.mp4) |
| raw768_native-seed43 | place_object_basket | 6202007 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/place_object_basket/seed-6202007.mp4) |
| svae48_native-seed43 | place_object_basket | 6202000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/place_object_basket/seed-6202000.mp4) |
| svae48_native-seed43 | place_object_basket | 6202002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/place_object_basket/seed-6202002.mp4) |
| svae48_native-seed43 | place_object_basket | 6202003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/place_object_basket/seed-6202003.mp4) |
| svae48_native-seed43 | place_object_basket | 6202006 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/place_object_basket/seed-6202006.mp4) |
| svae48_native-seed43 | place_object_basket | 6202007 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/place_object_basket/seed-6202007.mp4) |
| raw768_native-seed44 | place_object_basket | 6202000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/place_object_basket/seed-6202000.mp4) |
| raw768_native-seed44 | place_object_basket | 6202002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/place_object_basket/seed-6202002.mp4) |
| raw768_native-seed44 | place_object_basket | 6202003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/place_object_basket/seed-6202003.mp4) |
| raw768_native-seed44 | place_object_basket | 6202006 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/place_object_basket/seed-6202006.mp4) |
| raw768_native-seed44 | place_object_basket | 6202007 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/place_object_basket/seed-6202007.mp4) |
| svae48_native-seed44 | place_object_basket | 6202000 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/place_object_basket/seed-6202000.mp4) |
| svae48_native-seed44 | place_object_basket | 6202002 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/place_object_basket/seed-6202002.mp4) |
| svae48_native-seed44 | place_object_basket | 6202003 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/place_object_basket/seed-6202003.mp4) |
| svae48_native-seed44 | place_object_basket | 6202006 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/place_object_basket/seed-6202006.mp4) |
| svae48_native-seed44 | place_object_basket | 6202007 | False | False | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/place_object_basket/seed-6202007.mp4) |

Registration commit: `1b39844daada3fd81ff135805a16a888d72c0fc9`. Source model commit: `e983676b069150d0346fcd9e90fd8ebe8a840f13`.
The archive includes every paired clip, contact sheet, per-episode result and independent audit, registered provenance and the sampled visual review.

[Full-quality evidence bundle](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/complete-pilot-evidence.zip)
