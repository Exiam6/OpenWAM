# Resource-only handoff after seed42

Registered before any seed43/44 attempt. Science, scene order within each cell,
checkpoint endpoints, success criteria, code and maximum total GPU occupation
remain those in PLAN.md. The completed seed42 adjust control passes for both
arms; the independent audit confirms identical initial inputs and first8 actions.

Current lanes each hold three sequential cells on GPUs3/4. Other registered work
on GPUs1/2 is approaching completion. To use the authorized four GPUs when they
become available, finish seed42 on the original workers, then distribute the
four **unstarted** seed43/44 cells across GPUs1–4. This changes scheduling only.

Do not edit a running code snapshot. A supervisor may terminate an owned original
coordinator only in its waiting-capacity state for seed43, with completed seed42,
no active server/child in the completed record, admission sample count at most1,
and no seed43/44 output directory. That state provides at least10 seconds before
the coordinator could start its next cell. Check /proc identity and ownership,
send SIGTERM, and verify its final record; never terminate an active episode.
If the window is missed or any original cell fails, stop the handoff for diagnosis.
The expected coordinator termination record remains visible and is annotated as
a between-cell resource handoff, not relabelled as a successful experiment.

After both lanes safely stop, run raw43 on GPU1, compact43 on GPU2, raw44 on GPU3,
compact44 on GPU4. Three admission samples still require own-user occupants only,
>=32GiB free and utilization<=85%. Ports and coordinator-state files are unique.
The run directory and evaluation code remain identical; attempted cells are never
retried. Each new worker receives only one registered, unstarted checkpoint cell.

Budget is unchanged: each arm's original10800-second active budget minus its
actual seed42 occupation, split equally between its two remaining cells. Thus
the combined six-cell budget cannot exceed6 occupied GPU-hours. No extra budget
is granted by this handoff. The deadline and pause paths stay unchanged. If a
per-cell allocation proves insufficient, preserve partial output and diagnose;
do not extend it silently. L2 measured projection is5.643GPU-hours in total.

Before launch, retain the exact scheduling-only diff, SHA256 of the new driver
and supervisor, source commit, final original-lane records, allocated seconds,
launch commands and process identities. New workers reuse the same training-scene
smoke, historical control reproduction and per-scene checks. The native success
results combine by checkpoint/scene, never by worker or GPU assignment.
