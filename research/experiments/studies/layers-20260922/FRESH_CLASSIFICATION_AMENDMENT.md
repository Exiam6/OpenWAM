# Collection classification correction, before any candidate scoring

Original collector exit1 after two accepted scenes. Scene600002 raised the native
UnStableError during initial environment setup. Original worker wrongly labeled
all exceptions infrastructure_failure; native collect_data.py explicitly treats
UnStableError as an infeasible scene and proceeds to the next seed.

The original source761f53d, launcher/logs, three result.json files and exit1 remain
unchanged. V2 handles this specific typed exception as unstable_scene. All other
exceptions still stop the collection. The old600002attempt is retained with a
separate classification annotation in the new manifest, never rerun.600000/1 are
reused without execution, next new attempt is600003. Same first20within40 rule,
all attempts visible, no candidate scores opened or filtering changes after scores.
This implements the already specified expert-feasibility filter, not a search for
favorable policy scenes. Wall deadline remains15:25:23CDT, two hours from the
first original worker start; profile waiting does not extend this allocation.
