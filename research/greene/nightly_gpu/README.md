# Daily GPU admission

User instruction,2026-10-19: inspect all16GPUs on gpu-01 and gpu-02 continuously;
between01:00and09:30America/Chicago use cards with no users/processes and at least
30minutes of observed idleness. Prefer gpu-01 GPU1/2. This supersedes historical
nightly windows in older plans; it does not change a study's models, fixed endpoint,
failure records, cost cap, or authorize duplicate attempts.

The controller samples both hosts every10seconds, rotates the order of extra-card
selection, and prioritizes gpu-01 GPU1/2. A card needs zero utilization, no compute
process and less than256MiB used for1800seconds. Missing/error samples, changed
UUID/policy, clock reversal or a gap above25seconds reset the history. Sampling
cannot prove absence of sub-sample activity. Eligibility is never inferred from
a single idle snapshot. The entire registered job plus120seconds must fit before
09:30. DST boundaries use the America/Chicago timezone database.

Runtime:`/home/zifanz4/openwam-runtime/nightly-gpu`. The user service
`openwam-nightly-gpu.service` maintains `latest.json` and compressed daily
observation logs. `policy.json` stores the user's resource instruction and all16
GPU UUIDs. Changes require a tested version and service restart.

Ready experiments are explicit JSON descriptors in `jobs/`. Required fields:
`id`, `status=ready`, canonical `scientific_identity`, hard `max_seconds`, hashed
`authorization_evidence`, and `host_commands` for each actually prepared host.
Each host command supplies absolute `argv`, a CPU-only `admission_argv`, `cwd`,
`input_sha256` and `outputs_absent`. Use the experiment's real admission command;
a ready descriptor does not bypass its study budget or scientific gates. Remote
assets/environments must be prepared and verified before listing that host.

The single controller exclusively claims a job before submission, checks the GPU
again and records its complete idle evidence. The remote/local worker takes a
GPU-UUID lock, validates inputs and absent outputs, runs admission and checks
capacity immediately before launch. It exports CUDA_VISIBLE_DEVICES as UUID and
OPENWAM_ALLOCATION_END, OPENWAM_GPU_INDEX, OPENWAM_GPU_HOST. Independent systemd
units bound each job and kill only its own control group. The worker stops its
own process group if another GPU process appears, a pause marker appears, its
budget expires, or the window ends. Claims, uncertain submissions, failures and
partial outputs are preserved without automatic retries. Completed job IDs and
scientific identities are never claimed again.

Already running VAE seeds42/43 are protected identities, not pending jobs. Their
original bounded supervisors remain in charge. The resource pool creates no
filler workload and does not duplicate completed G/H experiments. An empty ready
queue is reported explicitly while all16cards continue to be observed.
