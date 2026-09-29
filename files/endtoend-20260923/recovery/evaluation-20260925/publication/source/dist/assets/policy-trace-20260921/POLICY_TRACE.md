# Optional policy request/action traces

Use this diagnostic to compare repeated runs with the same scenes, prompts, ordering
and settings. It does not change default evaluation or import the model/torch stack.
Wrap the existing transport in an integration you control:

```python
from benchmarks.utils import WSPolicyClient
from benchmarks.utils.policy_trace import TracedPolicyClient

with TracedPolicyClient(WSPolicyClient("ws://127.0.0.1:8848"), "repeat0.jsonl",
                        metadata={"scene_seed": 400000, "protocol": "my-fixed-protocol"}) as client:
    client.reset()
    response = client.predict(payload)  # payload comes from your existing observation loop
```

The wrapper delegates exactly once and returns the original response object.
`predict_once` preserves the underlying no-retry method; `predict` preserves its
existing retry behavior. Reset calls and failed calls are recorded. Closing the
wrapper closes the transport. Paths must be new; I/O/serialization errors stop the
caller rather than silently dropping trace records. Use separate files for repeats.

Compare without loading a model:

```bash
python -m benchmarks.utils.policy_trace repeat0.jsonl repeat1.jsonl
```

Results give the first differing **event index** per field, plus event counts and
metadata agreement. Reset/error events also occupy indices. Different lengths are
not identical traces even when their shared prefix matches. Match metadata and
inspect server counters before attributing a difference. No automatic scene pairing
or claim of determinism follows from equality in a finite run.

Hashes cover the submitted payload, encoded image bytes, prompt, state and returned
action. State/action values and server counters are saved; image pixels and prompt
text are not. Encoded-image equality is not a numerical pixel-difference measure.
These are **server actions**, not executed joint motion. In RoboTwin the native
`joint_action.vector` contains drive targets; use actual qpos/qvel getters for physics
diagnostics. Identical task outcomes need not imply identical traces. A changed
request before a changed action narrows investigation, but does not identify a cause:
action buffers may execute an older generated chunk. Tracing adds CPU/I/O overhead
and does not prove that traced and untraced execution have identical timing.
