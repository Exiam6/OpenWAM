# Five-episode smoke runs write a denominator of 100

Verified against RoboTwin commit0aeea2d669c0f8516f4d5785f0aa33ba812c14b4 and OpenWAM commit7c5861e45cfe1339a0323f0e0b03a3316c37971c. OpenWAM's wrapper sets the `eval_policy(..., test_num=5)` argument through ROBOTWIN_TEST_NUM. The caller `RoboTwin/script/eval_policy.py:main` retains its local `test_num=100`, so its `_result.txt` writer divides the successes by100. The rollout progress log uses the real denominator and is correct.

CPU reproduction executes the actual upstream main/reporting function with a stub rollout, using the actual OpenWAM cap wrapper. All5 successful stub rollouts produce0.05 in the original result file; after the proposed RoboTwin-side patch the value is1.0. The uncapped100-rollout default remains1.0 when all100 succeed. This reproduction does not claim real robot successes. See robotwin-smoke-denominator-reproduction.json and scripts/reproduce_smoke_denominator.py.

Minimal patch: let RoboTwin main read and validate ROBOTWIN_TEST_NUM, so evaluation and its writer share the same count. `git apply --check` succeeds on the pinned RoboTwin checkout. Patch is prepared separately and was NOT applied to the running smoke test. Its code and outputs remain auditable; our report reads actual numerator/denominator from rollout logs and preserves the original result file.

This is a RoboTwin/OpenWAM evaluation-integration issue, not a representation-model defect. Prefer a small RoboTwin PR plus an OpenWAM compatibility note after maintainer review; no upstream PR or message has been sent.
