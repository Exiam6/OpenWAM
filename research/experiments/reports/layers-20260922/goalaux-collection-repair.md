# Native collector portability repair

An isolated copy of the native environment passed empty-scene rendering and GPU
binding checks. At the first candidate, two robot YAML files still referenced the
old filesystem root. Missing collision files activated the wrapper's fallback
planner. The added actual-backend assertion stopped initialization before any
expert action, saved trajectory, or confirmation score. Original failure retained.

The prospective scientific protocol permits documented repairs of failed work.
The continuation replaces only the two copied absolute asset path prefixes;
original robot URDF and collision geometry hashes match. Physics, camera, planner,
rendering, seeds, acceptance rules and collection deadline remain unchanged.
The interrupted initialization is repaired at the same candidate; no completed
expert-feasibility attempt is repeated or discarded. Both actual planner classes
and rendering device are checked on every new scene. This is an environment
portability finding, not evidence that the representation improves control.
