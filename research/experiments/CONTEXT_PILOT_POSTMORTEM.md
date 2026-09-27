# Context pilot ended early — 2026-09-22

A0 resident_idle completed two fixed commands. B0 passed all3 GPU idle checks
but failed the plain socket.bind port18848 check with Errno98 Address already
in use. B0 never started a server/client; B1 and A1 were not run. Inner and outer
exit codes1. No restart, protocol amendment, extra trial or seed replacement.
One completed trial cannot answer the predeclared A/B comparison.

Started outer00:28:26.608CDT; internal timer00:28:42.299 after source verification;
ended00:31:40.936, internal178.636seconds. A0 server ready104.627seconds; total
trial150.922seconds. Cleanup targeted only own dedicated process groups; unrelated
exp tmux retained. GPU returned6MiB/0%/0ECC. All work ended within600second cap.

Post-run audit:55 runtime source hashes including checkpoint,6 protocol/input
hashes unchanged; all2 requests,6 decoded images,4 planner records and saved
20D/16D commands verified. One video verified2 frames. A0 transport events are
reset,reset,selection,selection; zero online predictions. No newly generated
model actions were executed in this pilot. No new success-rate estimate.

CPU diagnosis reproduced a possible engineering mechanism on an EPHEMERAL
loopback port: after active connection close, socket stateTIME_WAIT with no
listener; plain bind givesErrno98, SO_REUSEADDR bind succeeds. Original18848
had no socket entries when checked later. Failure-time socket state was NOT
captured, so do not conclusively attribute the original error toTIME_WAIT or
claim some other user's listener caused it. No server/socket settings changed
for the frozen pilot. DiagnosticJSON retains exact observations.

Follow-up: retain this incomplete pilot and all planned rows. Port preflight
should eventually distinguish actual listening sockets from reusableTIME_WAIT
and be covered by an appropriate lifecycle test, independently of this frozen
run. Do not retry this scientific pilot automatically. The planned comparisons
are invalid/missing, not negative evidence about online inference. Per stopping
rule, stop expanding simulator debugging here and return to representation/
robustness contribution packaging and honest claims/held-out validation planning.
