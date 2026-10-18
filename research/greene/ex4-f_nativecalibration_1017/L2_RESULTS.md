# Restored calibration improves the seed42 adjust pilot

On the fixed first five accepted adjust-bottle scenes, both endpoint policies improve after restoring native planner calibration. This is one training seed and one task; handover, place and the other training seeds remain pending.

Next: finish the registered three-seed pilot under the unchanged compute cap. Do not interpret this control-task result as a representation ranking.

| Model | Old path | Restored native calibration | Paired gains | Paired losses |
|---|---:|---:|---:|---:|
| raw768_native | 40% (2/5) | 100% (5/5) | 3 | 0 |
| svae48_native | 40% (2/5) | 100% (5/5) | 3 | 0 |

The background question is whether execution failure obscures learned-policy performance. The single change preserves the `aloha-agilex` filename marker that selects native arm-base calibration. Checkpoint, prompts, random seeds, initial scene, cameras, action horizon and success rule are held fixed. Initial proprioception, all three camera views and the first eight actions match the historical control exactly in all ten pairs.

The preregistered easy-task gate requires at least one success per model and no infrastructure failures before advancing. Both models pass. This gate establishes a working policy-executor combination; the primary handover hypothesis is still pending.

## Every paired clip

Old is above; restored calibration is below. The three original camera views are shown from left to right: head, left wrist, right wrist. Clips align action index at12fps, not physical time; successful shorter episodes hold their final frame. All clips fully decode. Selected start/intermediate/final contact sheets were inspected; this is not a claim of manual frame-by-frame review. No external full-robot camera was added.

| Model | Scene | Old | Restored | Video |
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

## Provenance and limits

- Registration commit: `1b39844daada3fd81ff135805a16a888d72c0fc9`; source model commit: `e983676b069150d0346fcd9e90fd8ebe8a840f13`.
- `data.json` contains every original and corrected result plus independent checks. `registration.json` contains scene/checkpoint/code hashes.
- Model names refer to raw768-dimensional versus compact48-dimensional frozen visual representations, with the same native action architecture.
- Development pilot: first-five selection rule fixed before inspecting this subset; original historical outcomes were available. No unseen-test or cross-seed claim.
- The native success rule is unchanged. Follow-up results must remain separate from this frozen control-task snapshot.

[Full-resolution evidence bundle](/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017/L2-adjust-seed42-evidence.zip)
