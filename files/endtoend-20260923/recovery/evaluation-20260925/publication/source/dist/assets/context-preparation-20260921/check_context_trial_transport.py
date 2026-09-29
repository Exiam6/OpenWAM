"""CPU gates only; no simulator, model, network or CUDA objects are constructed."""

import copy
import gzip
import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np
from context_trial_transport import ContextTrialTransport

CODE = Path("/home/zifanz4/openwam-trace-contribution")
sys.path.insert(0, str(CODE))

RESULT = Path("/data02/zifanz4/openwam-experiments/results")
SOURCE = RESULT / "repeatability-20260921/closedloop/r0-clean"
ROWS = [
    json.loads(line)
    for line in (SOURCE / "seed-400000-trace.jsonl").read_text().splitlines()[:2]
]
COMMANDS = [row["server_action"] for row in ROWS]
PAYLOAD = json.loads(gzip.decompress((SOURCE / ROWS[0]["request_file"]).read_bytes()))


class FakeClient:
    def __init__(self):
        self.calls = []
        self.response = {"action": [99.0] * 20, "step": 917, "latency_ms": 4.0}

    def predict(self, payload):
        raise AssertionError("retry-enabled predict must not be used")

    def predict_once(self, payload):
        self.calls.append(payload)
        return self.response

    def reset(self):
        return {"type": "reset_ack"}

    def ping(self):
        return {"type": "pong"}

    def close(self):
        pass


class ContextChecks(unittest.TestCase):
    def setup_transport(self, mode):
        client, records = FakeClient(), []
        transport = ContextTrialTransport(client, COMMANDS, mode, records.append)
        transport.reset()
        return client, records, transport

    def test_idle_has_no_predict_and_no_fabricated_server_counter(self):
        client, records, transport = self.setup_transport("resident_idle")
        for i in range(2):
            response = transport.predict(PAYLOAD)
            self.assertEqual(response["action"], COMMANDS[i])
            self.assertNotIn("step", response)
            self.assertIsNone(records[-1]["online_response"])
        self.assertEqual(client.calls, [])
        with self.assertRaises(RuntimeError):
            transport.predict(PAYLOAD)
        with self.assertRaises(RuntimeError):
            transport.reset()

    def test_shadow_calls_once_keeps_unexpected_response_and_saved_action(self):
        client, records, transport = self.setup_transport("shadow_online")
        response = transport.predict(PAYLOAD)
        self.assertEqual(response["action"], COMMANDS[0])
        self.assertEqual(len(client.calls), 1)
        self.assertIs(client.calls[0], PAYLOAD)
        self.assertEqual(records[-1]["online_response"]["step"], 917)
        self.assertFalse(records[-1]["online_action_equals_fixed"])
        response["action"][0] = -42
        client.response["action"][0] = -42
        self.assertEqual(records[-1]["selected_action"], COMMANDS[0])
        self.assertEqual(records[-1]["online_response"]["action"][0], 99)

    def test_ambiguous_failure_aborts_and_is_never_retried(self):
        client, records, transport = self.setup_transport("shadow_online")
        error = TimeoutError("ambiguous")

        def fail(payload):
            client.calls.append(payload)
            raise error

        client.predict_once = fail
        with self.assertRaises(TimeoutError) as caught:
            transport.predict(PAYLOAD)
        self.assertIs(caught.exception, error)
        with self.assertRaises(RuntimeError):
            transport.predict(PAYLOAD)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(records[-1]["event"], "error")
        self.assertEqual(transport.index, 0)

    def test_both_modes_preserve_inputs_and_rng(self):
        original = copy.deepcopy(PAYLOAD)
        for mode in ["resident_idle", "shadow_online"]:
            _, _, transport = self.setup_transport(mode)
            before_np, before_py = np.random.get_state(), random.getstate()
            transport.predict(PAYLOAD)
            self.assertEqual(PAYLOAD, original)
            self.assertTrue(
                all(
                    np.array_equal(a, b)
                    for a, b in zip(before_np, np.random.get_state())
                )
            )
            self.assertEqual(before_py, random.getstate())

    def test_rejects_invalid_configuration_and_use_before_reset(self):
        for commands, mode in [(COMMANDS[:1], "resident_idle"), (COMMANDS, "live")]:
            with self.assertRaises(ValueError):
                ContextTrialTransport(FakeClient(), commands, mode, lambda r: None)
        t = ContextTrialTransport(
            FakeClient(), COMMANDS, "resident_idle", lambda r: None
        )
        with self.assertRaises(RuntimeError):
            t.predict(PAYLOAD)
        t.close()
        with self.assertRaises(RuntimeError):
            t.reset()

    def test_actual_native_eval_has_identical_16d_targets_in_both_modes(self):
        from benchmarks.robotwin import openwam2robotwin_interface as iface

        observation = {
            "observation": {
                k: {"rgb": np.zeros((4, 4, 3), np.uint8)}
                for k in ["head_camera", "left_camera", "right_camera"]
            },
            "endpose": {
                "left_endpose": [0, 0, 0, 0, 0, 0, 1],
                "right_endpose": [1, 2, 3, 0, 0, 0, 1],
                "left_gripper": 1.0,
                "right_gripper": 1.0,
            },
        }

        class Env:
            take_action_cnt = 0

            def get_instruction(self):
                return "fixed instruction"

            def take_action(self, action, *, action_type):
                self.received.append((action.copy(), action_type))
                self.take_action_cnt += 1

        targets = []
        for mode in ["resident_idle", "shadow_online"]:
            _, _, t = self.setup_transport(mode)
            model = iface.ModelClient.__new__(iface.ModelClient)
            (
                model._send_state,
                model._state_dim,
                model._action_indices,
                model._action_type,
            ) = True, 20, None, "ee"
            model._task_description, model._debug, model._step, model._client = (
                "fixed instruction",
                False,
                0,
                t,
            )
            env = Env()
            env.received = []
            for _ in range(2):
                iface.eval(env, model, observation)
            targets.append(env.received)
        for i, (left, right) in enumerate(zip(*targets)):
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(
                left[0], np.asarray(ROWS[i]["env_action"], dtype=np.float32)
            )
            self.assertEqual(left[1], right[1])


if __name__ == "__main__":
    unittest.main()
