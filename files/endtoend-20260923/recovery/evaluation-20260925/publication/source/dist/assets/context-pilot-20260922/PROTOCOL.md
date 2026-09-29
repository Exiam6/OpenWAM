# Frozen context pilot — 2026-09-22

A NEW bounded study; no extension of the completed90min repeatability allocation.
Question: with identical executed commands and original model resident, does
online prediction/RPC/waiting alter first-post-action observation repeatability?
Implements CONTEXT_CONTROL_DESIGN.md without changing completed protocols.

Order exactly A0 resident_idle, B0 shadow_online, B1 shadow_online, A1 resident_idle.
Each trial has a new original-policy server and a new native simulator process.
Only clean firstscene400000; two commands each; eight commands maximum. Reuse
stageC round0-clean first two saved20D commands, native conversion16D unchanged.
Initial full request must match source. Idle sends no obs; shadow sends exactly
one no-retry obs per step, stores actual reply, executes fixed command regardless
of any valid action difference. Expected online counters1,2; no fake idle counter.
Normal native episode+instruction-change resets preserved. Model seed42, compile
off, original synchronous full32-action executor, same checkpoint and encoder.
Only pick_dual_bottles step limit overridden2 in these new client processes.
No model/adapter training, original80evals not repeated. Not a success-rate study.

Measurements: both arms same request/planner/physical-state tracing. Primary
full request/raw cameras/EEF at observation1 (after first fixed command).
Secondary: measured qpos/qvel/root/link/object state before/after action0 and
planner inputs/results. Compare A0/A1,B0/B1 within-arm first, then A0/B0,A1/B1.
Report all rows and actual online-action differences. Only descriptive results;
no p-values, global determinism, isolated GPU-load or root-cause claim.

New hard wall cap600seconds, work cutoff595 with5seconds reserved for cleanup.
One L40S: cm001 GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3, existing advisory lock.
Before each trial: three idle samples5seconds apart, memory<100MiB, util0, ECC0,
no compute process; port18848 free. Never stop another task to obtain resources.
Server startup limited180seconds AND global cutoff; any trial timeout/nonzero/
initial mismatch/counter error/malformed reply stops this pilot. Missing trials
are marked not run, not replaced. No attempt-dependent change, retry, new seeds,
new conditions or extension after timeout. Temporary mock child tests are not
research trials. One pilot only; after completion/failure return to representation
work rather than extending simulator debugging without a new scientific reason.

Supervisor records exact own Popen IDs/start ticks and uses dedicated sessions.
Cleanup signals only those owned groups, including their same-session children;
PID reuse refuses signalling. CPU tests cover early server exit, client nonzero,
startup/client timeouts, owned descendant cleanup and an unrelated live sentinel.
An external timeout600s provides a second stop path. CPU startup/cleanup gates6
pass, plus prior native-loop/WebSocket8 and transport6 gates. Complete assembled
client checked in both modes using the native eval/converter with fake environment.
These are engineering checks, not proof that model+simulation will complete.

Source/config/command hashes recorded before launch in source-manifest.json;
checkpoint SHA256 verified streaming against existing published download hash.
Inputs/assets unchanged; sources copied from committed private checkout into
runtime only when identical shared frozen files or new context files. Native
runtime sources pinned to earlier validated commits. Any source mismatch aborts.
All partial logs, nonzero exits and complete rows retained. Complete observations
saved before initial mismatch gate. Results go to new results/context-pilot-20260922;
no old output directory is reused. Publish all outcomes, including no divergence.
