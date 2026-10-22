# CPU-only independent sample supervision

Question: can an independent parent reject a hung observation worker within a finite deadline, preserving bounded raw output and killing only its dedicated child session, including children holding output pipes? This addresses the explicit blocking /proc-read limitation of baseline 1c872d0 (43 CPU checks), not the unresolved native GPU cause.

Intervention: a read-only command supervisor around a future fresh observer process, with monotonic deadline, byte limit, dedicated process group and bounded cleanup. No production monitor or dispatcher integration. Hypothesis: hanging child and descendant fixtures fail closed with bounded parent response. Competing explanation: kernel/spawn stalls or escaping descendants remain beyond this user-space guarantee.

Fixed: official source 782e3133 synchronized 2026-10-22T02:00:33UTC with no changes; native planner, inference, scene, checkpoints and all frozen scientific attempts unchanged. ex8-k remains sole pending diagnostic. No GPU invocation, reservation, new scientific identity or full-row release. Existing <=280GPUh/seven-day stage and every older cap unchanged.

Metrics and criterion: success only for a zero-exit complete JSON object before deadline, bounded combined stdout/stderr, malformed/empty/nonzero/timeout/oversize rejected. Actual sleeping worker and a descendant holding pipes must be stopped with no indefinite wait. Partial raw output is attached to failures; cleanup state recorded. Tests pass in Python3.10/3.12 and inherited43 tests remain passing. Budget zero GPU seconds, ten-minute takeover deadline. Stop on global pause, unresolved test failure or ex8-k outcome demanding redirection.

Limits to disclose: Popen/kernel scheduling cannot be absolutely bounded by this process; setsid-escaping children are outside containment; no continuous isolation proof or native scene validation. Caller must run only trusted frozen code under dispatcher's own hard cap/cgroup, persist evidence, and stop native operations on observation error.

Outputs: separately committed supervisor, CPU tests and findings. No deployment/experiment dispatch from CPU fixtures. Next audit ex8-k actual result, review integration, freeze a separate bounded native follow-up before any GPU execution.
