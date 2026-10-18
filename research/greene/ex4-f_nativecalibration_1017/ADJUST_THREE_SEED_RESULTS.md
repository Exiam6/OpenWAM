# Native calibration recovers the adjust-bottle pilot across three training seeds

The completed adjust-bottle control tests whether restoring native planner calibration improves an already working task. It covers the same five frozen scene IDs for each of two policies and three training seeds; this is a development subset, not 30 independent scene IDs.

The single execution change preserves the ALOHA configuration marker. Final step20000 checkpoints, native inference, prompts, RNG, initial scene, camera views and success rule remain paired. This comparison estimates the execution change within each model; it does not establish a representation advantage.

| Model | Training seed | Old execution | Native calibration | Gains | Losses |
|---|---:|---:|---:|---:|---:|
| raw768_native | 42 | 40% (2/5) | 100% (5/5) | 3 | 0 |
| raw768_native | 43 | 40% (2/5) | 100% (5/5) | 3 | 0 |
| raw768_native | 44 | 20% (1/5) | 100% (5/5) | 4 | 0 |
| svae48_native | 42 | 40% (2/5) | 100% (5/5) | 3 | 0 |
| svae48_native | 43 | 60% (3/5) | 100% (5/5) | 2 | 0 |
| svae48_native | 44 | 40% (2/5) | 100% (5/5) | 3 | 0 |

The preregistered seed42 easy-task gate requires at least one success per model with no infrastructure failure. Both models pass; seeds43/44 provide replication on the same development scenes. The primary handover hypothesis and the full pilot remain pending in this frozen snapshot. Next: complete handover and place, then use the separately registered corrected factorial to compare architectures.

All30 pairs pass independent checks: exact initial state, all three initial camera errors zero, finite action traces and identical first native action chunks. Every video fully decodes. Four newly inspected contact sheets cover start, intermediate and final samples for both models at seeds43/44; the prior seed42 review remains separate. These are sampled visual checks, not manual inspection of every frame.

Videos show the actual head and two wrist policy views, old above and corrected below. Native wrist images are often blank or occluded; the lifted bottle can approach the head-view boundary. No extra external full-robot view was recorded. Alignment uses action index at12fps, not physical time; a shorter episode holds its final frame.

## All paired clips

| Model and seed | Scene | Old outcome | Corrected outcome | Video |
|---|---:|---|---|---|
| raw768_native-seed42 | 6200001 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200001.mp4) |
| raw768_native-seed42 | 6200002 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200002.mp4) |
| raw768_native-seed42 | 6200003 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200003.mp4) |
| raw768_native-seed42 | 6200006 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200006.mp4) |
| raw768_native-seed42 | 6200007 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed42/adjust_bottle/seed-6200007.mp4) |
| svae48_native-seed42 | 6200001 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200001.mp4) |
| svae48_native-seed42 | 6200002 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200002.mp4) |
| svae48_native-seed42 | 6200003 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200003.mp4) |
| svae48_native-seed42 | 6200006 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200006.mp4) |
| svae48_native-seed42 | 6200007 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed42/adjust_bottle/seed-6200007.mp4) |
| raw768_native-seed43 | 6200001 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200001.mp4) |
| raw768_native-seed43 | 6200002 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200002.mp4) |
| raw768_native-seed43 | 6200003 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200003.mp4) |
| raw768_native-seed43 | 6200006 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200006.mp4) |
| raw768_native-seed43 | 6200007 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed43/adjust_bottle/seed-6200007.mp4) |
| svae48_native-seed43 | 6200001 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200001.mp4) |
| svae48_native-seed43 | 6200002 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200002.mp4) |
| svae48_native-seed43 | 6200003 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200003.mp4) |
| svae48_native-seed43 | 6200006 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200006.mp4) |
| svae48_native-seed43 | 6200007 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed43/adjust_bottle/seed-6200007.mp4) |
| raw768_native-seed44 | 6200001 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200001.mp4) |
| raw768_native-seed44 | 6200002 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200002.mp4) |
| raw768_native-seed44 | 6200003 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200003.mp4) |
| raw768_native-seed44 | 6200006 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200006.mp4) |
| raw768_native-seed44 | 6200007 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/raw768_native-seed44/adjust_bottle/seed-6200007.mp4) |
| svae48_native-seed44 | 6200001 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200001.mp4) |
| svae48_native-seed44 | 6200002 | success | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200002.mp4) |
| svae48_native-seed44 | 6200003 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200003.mp4) |
| svae48_native-seed44 | 6200006 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200006.mp4) |
| svae48_native-seed44 | 6200007 | failure | success | [Play / download](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/paired-videos/svae48_native-seed44/adjust_bottle/seed-6200007.mp4) |

## Provenance

Registration: `1b39844daada3fd81ff135805a16a888d72c0fc9`; model source: `e983676b069150d0346fcd9e90fd8ebe8a840f13`.
The bundle contains all30 paired clips, their contact sheets and metadata, per-episode results and audits, the registration, and the sampled visual review. Historical failed outcomes and the earlier seed42 evidence bundle are retained. Subsequent results do not alter this snapshot.

[Full-quality evidence bundle](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/adjust-three-seed-evidence.zip)
