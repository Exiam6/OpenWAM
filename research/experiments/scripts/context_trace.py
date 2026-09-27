"""Separate fixed-command records from genuine online replies; CPU-prepared only."""

import copy
import gzip
import json

import numpy as np
from attribution_trace import AttributionTrace
from context_trial_transport import ContextTrialTransport
from repeat_common import array_hash, payload_hash
from trajectory_trace import instrument_interface


class ContextTrace(AttributionTrace):
    def __init__(self, out, initial_payload_hashes, extract_state, mode):
        if mode not in {"resident_idle", "shadow_online"}:
            raise ValueError("invalid mode")
        super().__init__(out, initial_payload_hashes, extract_state)
        self.mode = mode
        self.events = self.out / "transport-events.jsonl"
        with self.events.open("x") as f:
            f.write(json.dumps({"schema": "context-transport-v1", "mode": mode}) + "\n")

    def emit(self, event):
        saved = copy.deepcopy(event)
        with self.events.open("a") as f:
            f.write(json.dumps(saved, allow_nan=False) + "\n")
        if saved["event"] == "selection":
            row = self.current
            if row is None or saved["fixed_command_index"] != row["step"]:
                raise RuntimeError("selection does not match active control step")
            if "selection" in row:
                raise RuntimeError("duplicate selection")
            row["selection"] = saved

    def begin(self, raw, altered, condition, seed, step):
        if condition != "clean" or seed != 400000 or step not in (0, 1):
            raise RuntimeError("trial outside fixed first-scene scope")
        super().begin(raw, altered, condition, seed, step)
        self.current.update(mode=self.mode, role="two-command context diagnostic")

    def predict(self, original, payload):
        row = self.current
        if row is None or row["requests"]:
            raise RuntimeError("missing or duplicate active request")
        sha = payload_hash(payload)
        path = self.out / "requests" / f"step-{row['step']:04d}.json.gz"
        path.parent.mkdir(exist_ok=True)
        body = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        with path.open("xb") as f:
            f.write(gzip.compress(body, mtime=0))
        row.update(
            request_sha256=sha, request_file=str(path.relative_to(self.out)), requests=1
        )
        # Save mismatched requests first, then abort; do not erase or replace them.
        if row["step"] == 0 and sha != self.initial[("clean", 400000)]:
            row["initial_request_matches"] = False
            raise RuntimeError("initial request mismatch")
        response = original(payload)
        selection = row["selection"]
        action = np.asarray(response["action"], dtype=np.float32)
        row.update(
            fixed_command_index=response["fixed_command_index"],
            selected_action=action.tolist(),
            selected_action_hash=array_hash(action),
            online_response=copy.deepcopy(selection["online_response"]),
        )
        if payload_hash(payload) != sha or selection["request_sha256"] != sha:
            raise RuntimeError("request mutation or mismatched selection")
        if (
            response["fixed_command_index"] != row["step"]
            or action.shape != (20,)
            or not np.isfinite(action).all()
        ):
            raise RuntimeError("invalid selected command")
        if not np.array_equal(
            action, np.asarray(selection["selected_action"], dtype=np.float32)
        ):
            raise RuntimeError("selected command differs from transport record")
        online = row["online_response"]
        if self.mode == "resident_idle":
            if online is not None or selection["online_calls"] != 0:
                raise RuntimeError("idle arm unexpectedly performed prediction")
            row["online_counter_valid"] = None
        else:
            row["online_counter_valid"] = (
                isinstance(online, dict) and online.get("step") == row["step"] + 1
            )
            if selection["online_calls"] != 1 or not row["online_counter_valid"]:
                raise RuntimeError("online prediction counter mismatch")
            live = np.asarray(online.get("action"), dtype=np.float32)
            if live.shape != (20,) or not np.isfinite(live).all():
                raise RuntimeError("malformed online action")
            # Valid differing actions remain recorded, never selected or excluded.
        return response


def bind_context_interface(interface, trace, commands, mode):
    """Use native get_model/eval. The caller still supplies native observations."""
    original_get, original_eval = interface.get_model, interface.eval

    def get_model(args):
        model = original_get(args)
        model._client = ContextTrialTransport(model._client, commands, mode, trace.emit)
        return model

    def evaluate(env, model, observation):
        trace.env = env
        return original_eval(env, model, observation)

    interface.get_model, interface.eval = get_model, evaluate
    instrument_interface(interface, trace)
