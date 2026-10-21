# OpenWAM quarter-hour self-check bot

The user's instruction on 2026-10-21 was to write a bot that self-checks every
15 minutes, after the original readiness pilot stopped and an empty queue was
left unattended overnight. This is operational supervision, not a new scientific
experiment or permission to change native evaluation settings.

`openwam-selfcheck.timer` runs at minute 00, 15, 30 and 45 of every hour.
The persistent user timer survives logout (user lingering is already enabled)
and catches up after downtime. The oneshot service has a 13-minute hard bound;
its flock and the systemd unit prevent overlapping takeovers. A failed run does
not disable future calendar activations.

Every tick reads the existing 16-GPU inventory and registered official
reproduction jobs, checks controller freshness, failure state, actual native-log
progress, pending dispatch and full-suite coverage. Historical failures with a
registered recovery remain in the report without provoking duplicate retries.
A fresh eligible idle card plus an undispatched job, a failed attempt without
recovery, an empty queue before the official endpoint, a stalled native log or
completed readiness without a full-stage plan requests a bounded Codex takeover.
The stopped/stale owned controller can be started/restarted directly; existing
GPU workers are separate units and are not restarted by this bot.

Codex uses the installed CLI and saved local ChatGPT authentication, the normal
workspace-write sandbox with automatic approval review, structured JSON output,
and a ten-minute timeout. No sandbox bypass option is used. The prompt requires
actual diagnosis, separate preregistered recoveries and CPU preparation whenever
possible. Unchanged issues that require a user decision remain visible in every
self-check without repeatedly invoking the agent. An agent's prose never counts
as a completed experiment: state is collected again after each invocation.

All GPU launches remain under the existing dispatcher: observed idle30min,
gpu01 indices1/2 prioritized at all hours, other cards only America/Chicago
01:00–09:30, and no preemption. Existing experiment and planning budgets remain
in force. The bot does not publish Sites content or send external messages.
Reports and alerts are local; they are not chat push notifications.

State directory: `/home/zifanz4/.local/state/openwam-selfcheck/`

- `latest.json`: full latest assessment, evidence snapshot, repairs and agent result.
- `heartbeat.json`, `history.jsonl`: concise current and historical checks.
- `alerts.jsonl`: local failures, takeover requests, controller repairs and results.
- `agent.json`, `runs/<time>/`: bounded takeover state, JSONL events, final JSON and stderr.
- `handoff.json`: takeover observations, actual actions and next step.
- `service.log`: scheduler's stdout/stderr (works without journal access).

Inspect or operate:

```
systemctl --user list-timers openwam-selfcheck.timer
systemctl --user start openwam-selfcheck.service
python3 research/greene/selfcheck_bot/bot.py --check-only
python3 research/greene/selfcheck_bot/bot.py --force-agent
systemctl --user stop openwam-selfcheck.timer
```

The existing shared pause file `state/paused.json` prevents the bot from making
changes and is also respected by GPU jobs. Do not create it merely to stop the
bot timer: it intentionally pauses the research dispatcher jobs too.

Validation: regression tests cover the exact overnight failure/empty queue,
known recovery, valid/stale/foreign GPU admission, the daytime policy,
per-job stalling despite progress elsewhere, pause, failed takeover continuity,
user-input deduplication, and invalid authorization. CLI smoke test read the
real AGENTS.md using a real shell command. Deployment also validates the systemd
calendar, units and a real background takeover in the systemd environment.

Reference: https://learn.chatgpt.com/docs/non-interactive-mode
