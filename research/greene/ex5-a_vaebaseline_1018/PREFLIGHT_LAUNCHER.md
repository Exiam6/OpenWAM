# Prepared preflight supervision

Status: CPU implementation only. No gpu-registration.json exists and no GPU job
has been admitted. The parent requires a concrete GPU amendment and frozen code,
completed G pilots, and the completed positive H control with its video bundle.

The proposed path uses one truly idle extra GPU from0/5/6/7 during the authorized
extra-GPU window. Three extra GPUs must be idle at each of three admission polls
and again immediately before launch, because the complete endpoint requires
three parallel training and evaluation jobs later. This is an availability check,
not a promise that other users will leave those GPUs free. The remaining window
must fit the full worst-case8.5-hour serial/parallel chain plus cleanup reserves.

The parent counts entire child lifetimes, including imports, asset checks, model
loading, inference and cleanup. It runs cache (7200s), then the3-step smoke,
native deployment contract and512-step tiny fit under a combined1800s clock.
A9000s aggregate preflight alarm reserves10s for termination. Children have their
own stricter stage guards. On expiry or pause it terminates only the subprocess
session it created; a failed child is never retried. Each stage records the exact
command, PID, exit status and occupied time. Failed artifacts stay in place.

After successful tiny fit, the slowest of the last eight32-step timing intervals
projects20000steps, with a900s planning reserve for construction, two development
loss passes and checkpoint I/O. That projection must fit10740s. The remaining
allocation must still fit two parallel3-hour phases. These are feasibility gates,
not guaranteed bounds. Passing preflight does not itself launch full training;
the final release still requires review of actual evidence and frozen evaluation
implementation.

CPU tests execute real child processes and check successful completion, preserved
nonzero exit without retry, timeout cleanup including a grandchild, and refusal
before spawn for pause/expired deadline. They do not exercise CUDA, native model
forwards, simulator evaluation or real resource admission. The first timeout test
had a /proc observation race after a child disappeared; the test now treats an
absent process as terminal. The unchecked initial audit was explicitly retracted
in preflight-launcher-static-audit-correction.json; use the newer validated audit.
