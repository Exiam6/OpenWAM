# Pair instruction generation with the simulator seed

RoboTwin's Base_Task seeds NumPy and Torch at setup, but Python `random.seed` is
commented out. Its language generator uses Python `random.shuffle` and
`random.choice`, so identical scene seeds need not receive identical language.
That confounds paired observation-corruption or policy-ablation comparisons.

The proposed patch scopes Python random to the actual `now_seed` only during
instruction-list generation and restores its prior state afterward. The existing
NumPy instruction selection then uses its unchanged, scene-seeded RNG. Scene and
expert-feasibility behavior is unchanged. This intentionally makes instruction
sampling deterministic per scene; it does not fix other simulator nondeterminism.

The standalone CPU reproduction executes the actual upstream generator and the
exact patched AST block: five distinct initial Python RNG states produce five
original lists but one patched list; outer RNG state is preserved. Patch applies
cleanly to RoboTwin 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4. The current running
benchmark checkout remains unmodified; our research wrapper applies equivalent
scoping. No PR has been sent.

Reproduce:

    python scripts/reproduce_instruction_seed.py --robotwin /path/to/RoboTwin --output /tmp/instruction-check
