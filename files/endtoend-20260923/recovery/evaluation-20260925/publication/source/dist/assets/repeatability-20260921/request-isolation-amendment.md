# Request-isolation correction before paired inference outcomes

The first processA generated one chunk, then the payload-integrity assertion failed:
PolicyServer.predict -> ObsPreprocessor.preprocess mutates its input dictionary,
adding a PIL image and converting state to ndarray. Native WebSocket handling
JSON-decodes every request. Reusing one dictionary in a direct diagnostic caller
would differ from that transport behavior. The early guard caught this before a
second request or any repeated-input/candidate comparison.

The complete failed process/log/source freeze/one generated chunk/exit1 are retained
in engineering-attempt-json-mutation and excluded as one invalid engineering run.
The six saved JSON requests and zero-action capture remain unchanged and verified.
The correction JSON-decodes a fresh copy for each server.predict call; a new CPU
check uses the actual upstream ObsPreprocessor twice and verifies identical image/
state outputs while the canonical JSON payload remains byte-hash unchanged.

Repeat processA and B under the original54-chunk sequence. No change to scenes,
noise, model, arms, seeds, numerical settings, analysis or comparison selection.
The90-minute allocation ceiling is measured from the original launch; resume gets
only the remaining time. No old normalization results or running jobs were touched.

An intermediate restart was stopped during model loading (zero chunks) after a
new CPU test reported an RNG assertion spanning lazy imports. The calling shell
had returned a later syntax check's zero status, leaving a stale previous check
file. This is retained in engineering-attempt-cpu-gate, including its misleading
zero exit marker without any completion artifact. The tracing RNG check is now
scoped immediately around tracing, the native preprocessing isolation check is
separate, and eight fresh CPU checks pass. The resume launcher reruns these checks
with fail-fast behavior and explicit termination exits before loading any model.
