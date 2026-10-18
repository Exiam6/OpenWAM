# Prepared deployment contract

Status: implementation prepared, GPU execution not admitted or performed.
The existing CPU-only registration is unchanged. This adds a concrete check to
the proposed serialization/cache-deployment gate in GPU_STAGE_DRAFT.md.

After the three-step seed42 smoke writes a checkpoint, a new process must load it
through the unmodified native deployment loader. On the first training window of
each of the three tasks, it compares raw-state deployment normalization against
the frozen training cache (maximum absolute error at most1e-6), the raw RGB first
frame's encoded latent against the cached causal first token (at most1e-6), and
all32 generated20D actions from native image versus cached-token inference
(at most1e-5). A repeated image inference at the same fixed seed must agree to
1e-5. All values must be finite. Fixed native10-step inference, shift5, input
resolution384x320 and9video/33action frames are retained. It preserves all three
action traces and source/checkpoint hashes, including failed attempts.

This is not a closed-loop success test. BF16 deployment casts FP32 trainable
master weights; the check does not assert bitwise FP32-master preservation.
No development outcome chooses the inputs or thresholds.

The future parent launcher must charge smoke, deployment contract and tiny fit
against the same1800-second allowance, including model loading and cleanup.
Each child checks the parent's remaining allowance and prior result hashes.
Tiny fit now requires the actual successful deployment result and its occupied
time. Full training requires successful deployment and tiny-fit artifacts whose
hashes match the release, in addition to the measured budget gate.

Admission remains closed when gpu-registration.json is missing. The CPU audit at
`/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/native-contract-static-audit.json`
verified that deployment, tiny fit and full training refuse this state without
creating run outputs. It did not exercise GPU inference or prove any runtime
threshold. A concrete GPU amendment, frozen launcher and completed predecessor
evidence are still required before execution.
