# CPU-only serialized observer contract

Question: can the prepared independent sample supervisor and observer safely exchange a complete, fresh, phase-bound inventory across JSON serialization? Baseline f21c456 has 59 CPU checks per Python but its generic supervisor accepts arbitrary JSON objects; it does not certify the observer schema or reject serialized own-peer contexts.

Single intervention: a CPU-only validation boundary that reconstructs the inventory from bounded raw queries and serialized trusted identities, checks host/map/ownership and nested monotonic query/worker/parent intervals, and rejects stale, pre-phase, incomplete, inconsistent or peer-context samples. Hypothesis: hostile/malformed/replayed packets are rejected while a real fresh observer-to-supervisor subprocess roundtrip is accepted. Competing explanation: field validation alone can mistake internally inconsistent summaries for a complete safe sample; raw reconstruction and cross-check address that failure.

Held fixed: observer/supervisor source, native code, planner/inference, all frozen runtimes, checkpoint, scenes, seeds, GPU queues/claims and all caps. Official sync 2026-10-22T02:15:32UTC succeeded unchanged 782e3133; no experimental upgrade. No scientific GPU attempt or registration.

Criteria fixed before checks: accepted complete fresh empty/prewarm sample and assigned-only sample; reject own-peer, foreign assigned, wrong host/map/unit, malformed identities, wrong command/query return, invalid or out-of-order/nonfinite timestamps, stale/pre-phase data and summary/raw disagreement. Failures retain the complete already bounded packet. Actual subprocess roundtrip and inherited 59 tests pass in system Python3.12 and benchmark Python3.10. Zero GPU seconds; finish within this ten-minute takeover. Stop on global pause or unresolved test failure.

Scope: trusted frozen worker code only. JSON cannot authenticate a hostile worker. No continuous isolation proof, no PID identity before first query, no atomic pair of nvidia-smi queries, no native host/cgroup validation. Worker cannot self-report its way around the parent timeout; dispatcher containment remains needed. No production integration or full-row release.

Deliverables: committed plan before tests, validator, adversarial CPU fixtures and findings with remaining native L1/L2/L3 requirements. ex8-k remains sole pending immutable probe; await actual outcome before freezing any GPU follow-up.
