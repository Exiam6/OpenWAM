# Recorded collection classifier repair, before any confirmation score

At fixed basket seed802008 the native first grasp returned no candidate pose.
The source path is choose_grasp_pose -> (None,None) -> grasp_actor's equality
branch -> Action(move,None) assertion. The exact traceback identifies the first
basket grasp at line73, grasp_actor at line1205, and Action at line76. Real
CuroboPlanner and the allocated GPU were recorded. The original result and both
collector/dependent-job exit1 remain unchanged. No trajectory/HDF5 was produced.

A CPU check executed the unchanged native grasp_actor AST and native Action
assertion with the null sentinel: it reaches this exact error. Valid branches and
the prior failed-plan short circuit remain unchanged. Negative classifier checks
reject other tasks, action locations, errors, nonzero exits and accepted episodes.

The scientific protocol already defines first20expert-feasible scenes and permits
a recorded repair of failed work. This amendment repairs only the collector's
error classification; no simulator/planner/task source or scientific criterion
changes. The source-proven exact null-initial-grasp class is retained as an
expert-infeasible candidate, with original_outcome=infrastructure_failure and an
explicit annotation. Unknown assertions remain infrastructure failures and stop.
No learned output, representation score or desired result informed this decision.

Do not rerun seed802008 or any completed attempt. Preserve all54attempts and47
accepted scenes from fresh-v2, resume the existing consecutive sequence at802009,
and keep at most40distinct candidates/task. New files use fresh-v3; accepted data
are referenced in place. Collection still ends2026-09-22T19:15:55.548938-05:00;
overall study still ends2026-09-23T01:15:12-05:00. If insufficient scenes remain,
confirmation is incomplete; no seed replacement or partial confirmatory scoring.
The same annotation applies prospectively to this exact source/traceback class.

Downstream numerical scorer/readouts/weights/thresholds are unchanged. Only input
and output directory routing changes to the continued manifest. The failed
waiting job did not execute identity/parity/extraction/scoring, so restarting only
its unstarted stages in confirmation-v2 does not repeat a completed evaluation.
Original scientific protocol and original source freezes remain immutable.
