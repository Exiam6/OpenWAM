# Resource probe correction before VAE execution

Observed before any VAE GPU attempt: the driver reports637MiB reserved memory
on otherwise idle GPUs. The original resource parents used total minus free
memory, which includes that reservation. This does not implement the registered
under256MiB-used criterion and can prevent all dedicated admission indefinitely.

This amendment queries memory.used and records memory.reserved separately.
No resident compute processes, zero utilization, under256MiB actual used,
physical UUID checks, three consecutive polls and a final check remain required.
The existing three-extra-GPU preflight rule and06:00-15:00UTC window remain.
No budget, scene, seed, training step, metric or numerical tolerance changes.

Freeze resource_probe.py, resource_preflight.py, resource_batch.py and
resource_schedule.py separately under resource-code-v2. Record their hashes,
this plan and CPU tests in resource-registration-v2.json, binding the unchanged
original gpu-registration.json hash. Keep all37original GPU runtime files.
The wrappers import the original parents and replace only their CPU eligibility
query functions. Original admission checks, asset checks, locks, accounting,
stage commands, cleanup and result paths execute unchanged. The cache, model,
training, native deployment, tiny fit and evaluation scripts remain byte-identical.

Retire the original idle CPU scheduler only before05:59UTC after verifying its
owner, command, no child and no preflight-run.json. Preserve its state and record
the intentional replacement. Start exactly one v2scheduler using the same lock,
with separate schedule state/log paths. Its sole GPU launch still occurs no
earlier than06:00UTC and uses the original full-endpoint window/gate logic.
This is not a scientific retry; no original VAE GPU attempt has started.

After actual preflight gates pass, use the separately frozen resource_batch.py
for training and evaluation batches. Full-training and per-seed evaluation
release generators remain the original frozen scripts. Reports must retain this
resource amendment and its provenance alongside the original registration.
