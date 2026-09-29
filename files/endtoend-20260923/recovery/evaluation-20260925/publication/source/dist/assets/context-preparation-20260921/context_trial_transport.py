"""CPU-prepared transport for a future two-action context probe; not launched.

Both modes return the same saved commands. shadow_online additionally performs
one no-retry real prediction and records its complete response without executing
it. resident_idle leaves the policy resident without predict calls. The contrast
therefore includes compute, RPC and waiting time; it does not isolate GPU load.
"""

import copy
import hashlib
import json
import math
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ContextTrialTransport:
    def __init__(self, client, commands, mode, emit):
        if mode not in {"resident_idle", "shadow_online"}:
            raise ValueError("unknown context mode")
        if len(commands) != 2 or any(
            len(a) != 20 or not all(math.isfinite(x) for x in a) for a in commands
        ):
            raise ValueError("exactly two finite 20D saved commands required")
        self.client, self.mode, self.emit = client, mode, emit
        self.commands = copy.deepcopy(commands)
        self.index, self.resets, self.failed, self.closed = 0, 0, False, False

    def _available(self):
        if self.closed or self.failed:
            raise RuntimeError("context trial is closed or failed")

    def reset(self):
        self._available()
        if self.index:
            raise RuntimeError(
                "fresh process required; cannot restart completed commands"
            )
        try:
            response = self.client.reset()
            self.emit({"event": "reset", "response": copy.deepcopy(response)})
            if response.get("type") != "reset_ack":
                raise RuntimeError("invalid reset acknowledgement")
        except Exception:
            self.failed = True
            raise
        self.resets += 1
        return response

    def predict(self, payload):
        self._available()
        if not self.resets or self.index >= 2:
            raise RuntimeError("reset required and at most two commands permitted")
        started = time.perf_counter_ns()
        row = {
            "event": "selection",
            "mode": self.mode,
            "fixed_command_index": self.index,
            "request_sha256": hashlib.sha256(canonical(payload).encode()).hexdigest(),
            "online_response": None,
            "online_calls": 0,
        }
        try:
            if self.mode == "shadow_online":
                row["online_calls"] = 1
                row["online_response"] = copy.deepcopy(
                    self.client.predict_once(payload)
                )
                # Keep unexpected values/counters for diagnosis, never select them.
                canonical(row["online_response"])
            row["roundtrip_ns"] = time.perf_counter_ns() - started
            row["selected_action"] = copy.deepcopy(self.commands[self.index])
            row["online_action_equals_fixed"] = (
                canonical(row["online_response"].get("action"))
                == canonical(row["selected_action"])
                if isinstance(row["online_response"], dict)
                else None
            )
            if (
                hashlib.sha256(canonical(payload).encode()).hexdigest()
                != row["request_sha256"]
            ):
                raise RuntimeError("transport mutated request")
            self.emit(copy.deepcopy(row))
        except Exception as error:
            self.failed = True
            self.emit({**row, "event": "error", "error_type": type(error).__name__})
            raise
        result = {
            "action": copy.deepcopy(self.commands[self.index]),
            "fixed_command_index": self.index,
        }
        self.index += 1
        # There is intentionally no synthetic server 'step'. Real counters live
        # only in online_response; the idle arm has no observed prediction step.
        return result

    def ping(self):
        self._available()
        return self.client.ping()

    def close(self):
        self.closed = True
        self.client.close()
