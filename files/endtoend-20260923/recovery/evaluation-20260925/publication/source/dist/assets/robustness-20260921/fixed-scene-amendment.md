2026-09-21, before any adapted-policy rollout: the per-condition expert filter
accepted seed 300002 in the original clean confirmation but skipped it in the
noise condition, before observation corruption was applied. No environment
exception was logged. This is evidence that repeated feasibility filtering does
not reliably produce the same accepted-seed list; its underlying cause is not
isolated. The initial confirmation attempt is retained under
invalid-refiltering-attempt and excluded from paired comparisons.

Freeze the first reference run's five expert-feasible seeds (300001–300005),
original generated unseen instructions, and initial image/proprio hashes in
fixed-scenes.json. Selection uses the expert-acceptance sequence, independent
of policy success. All four confirmation arms are rerun on these exact scenes.
The official rollout loop is retained via an audited AST transformation that
only disables repeated expert filtering and supplies the frozen seed per loop.
Replay the reference prompt while retaining the original NumPy choice length
and random-number consumption. Before the first action in every episode, require
exact reference image and proprio hashes. Setup failures are fatal, not skipped.

This changes the originally planned per-condition filtering protocol and is
explicitly reported. It does not change the model, noise, selected checkpoint,
confirmation scene range, action budget or outcome metric. The 15 baseline
rollouts remain valid: their actual scene/prompt/image/state pairs already match.
