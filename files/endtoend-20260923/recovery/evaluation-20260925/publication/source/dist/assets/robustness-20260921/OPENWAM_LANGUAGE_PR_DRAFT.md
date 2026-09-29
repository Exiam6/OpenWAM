# Add opt-in scene-seeded language for paired RoboTwin evaluations

When two evaluations reuse a RoboTwin scene seed, they can still receive different
instructions: RoboTwin seeds NumPy and Torch, but its instruction generator also
uses Python random.shuffle/choice. This confounds paired policy or observation
ablations. Add ROBOTWIN_SCENE_SEEDED_LANGUAGE=1 to scope language sampling to the
actual simulator seed and restore the caller's Python RNG afterward. The option
is off by default, preserving existing evaluation behavior and paper settings.

Validation: six CPU tests cover same-scene reproducibility, distinct scenes,
argument forwarding, RNG restoration on success/failure, missing seed errors,
and default-off/bootstrap behavior. Ruff lint and format checks pass. Against the
real RoboTwin generator at 0aeea2d, five different outer RNG states yield identical
instruction lists with this option, matching the separately tested upstream fix.
The research rollout wrapper uses the equivalent scoped operation; the new
OpenWAM patch itself has not been rerun end-to-end with SAPIEN.

Branch: contribution/robotwin-paired-language. Three files, 134 inserted lines,
including README and tests; no model changes or added dependencies. This is a
local review draft, not an opened or sent pull request.
