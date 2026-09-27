# Prompt cache eviction failure — isolated candidate repair

Observed Sep26 2026 in the frozen local OpenWAM deployment. Three closed-loop queues initially failed after45 completed episodes; the failing call is `_BoundedPromptEmbedCache.__setitem__ → popitem(last=False) → __getitem__ → move_to_end`, raising KeyError. This is infrastructure failure, not task failure or a representation comparison.

A standalone extraction of the exact class reproduces the exception on distinct insertion33 at capacity32 in the experiment Python3.10.12. The minimal patch removes the oldest key directly, avoiding re-entry through the LRU read override. Four capacities and20000 randomized operations check bounded size, ordering, value identity, repeated writes, cache hits, missing keys and clear. All pass. Full model inference with the fix has NOT been run.

The original local source and both isolated study copies have the same engine SHA256 (recorded in regression.json). These source directories lack Git metadata; upstream commit provenance has not been independently verified, so this is a candidate project bug fix, not a claimed submitted or accepted upstream contribution.

No frozen source was changed, failed episode replayed, deadline extended, or process terminated. Raw failed/incomplete evidence remains intact. A future recovery must be independently registered with its source version, limits and treatment of partial results before launch. Do not mix fixed and unfixed results as an unchanged full matrix.

Reproduce using `check_cache.py SOURCE_ENGINE_PATH OUTPUT_JSON`; the script extracts only the cache class and imports no model code. `engine.patch` targets `openwam/deploy/engine.py`. No GPU is required.
