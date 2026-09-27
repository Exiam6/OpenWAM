# Recorded repair of failed initialization, before any expert trajectory

The original collector stopped at seed800000 during setup, before play_once,
trajectory planning, replay, saved HDF5, or any learned scoring. Both planner
objects were fallback MplibPlanner because two copied curobo config files still
pointed at unavailable Group1 collision assets. The PCI binding check passed.
The failure/result/exit1 and original source freeze remain unchanged under fresh/.

The prospective MATCHED_PROTOCOL explicitly permits a documented repair of
failed work. This continuation repairs that uncompleted initialization; it does
not replay any completed expert trajectory, accepted/rejected feasibility result,
training fit or evaluation. It keeps seed800000 as the first candidate instead
of silently discarding it for an infrastructure fault. The source program checks
that there was exactly one prior setup failure and no trajectory/HDF5/plan result.
All new files go under fresh-v2/; the failed first setup stays in its lineage.

Only the two aloha-agilex YAML absolute asset prefixes were relocated, with an
exact reversible textual mapping. Original YAML, URDF and collision geometry
hashes were verified; the URDF and both collision files are unchanged. No task,
robot, planner, physics, camera, rendering, objective, seed sequence or acceptance
criterion changes. Per-scene real CuroboPlanner/PCI assertions remain active.

The original collection deadline 2026-09-22T19:15:55.548938-05:00 and overall
2026-09-23T01:15:12-05:00 deadline are retained. Max40distinct candidate seeds per
task; first20expert-feasible complete scenes. One extra failed setup is explicitly
reported, never counted as a fresh scene or evidence of performance. Further
infrastructure failures stop again. No selection based on confirmation outcomes.
