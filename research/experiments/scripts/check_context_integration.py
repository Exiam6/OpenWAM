"""CPU native-loop/recording and loopback-WebSocket gates, no model or simulator."""

import ast
import contextlib
import importlib
import io
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

import numpy as np
from capture_repeat_inputs import capture_payload, make_recording_model
from context_trace import ContextTrace, bind_context_interface
from context_trial_transport import ContextTrialTransport
from norm_client import install_replay
from repeat_common import payload_hash

RUNTIME = Path("/data02/zifanz4/openwam-experiments")
sys.path.insert(0, "/home/zifanz4/openwam-trace-contribution")
iface = importlib.import_module("benchmarks.robotwin.openwam2robotwin_interface")
SOURCE = RUNTIME / "benchmarks/RoboTwin/script/eval_policy.py"
ROWS = [
    json.loads(s)
    for s in (
        RUNTIME
        / "results/repeatability-20260921/closedloop/r0-clean/seed-400000-trace.jsonl"
    )
    .read_text()
    .splitlines()[:2]
]
COMMANDS = [r["server_action"] for r in ROWS]


def observation():
    return {
        "observation": {
            k: {"rgb": np.zeros((4, 5, 3), np.uint8)}
            for k in ["head_camera", "left_camera", "right_camera"]
        },
        "endpose": {
            "left_endpose": [0, 0, 0, 0, 0, 0, 1],
            "right_endpose": [1, 2, 3, 0, 0, 0, 1],
            "left_gripper": 1.0,
            "right_gripper": 1.0,
        },
        "joint_action": {"vector": np.zeros(6)},
    }


class Pose:
    def __init__(self):
        self.p = np.zeros(3, np.float32)
        self.q = np.array([1.0, 0, 0, 0], np.float32)


class Entity:
    def __init__(self):
        self.qpos = np.array([0.123, 0, 0], np.float32)
        self.qvel = np.zeros(3, np.float32)
        self.global_pose = Pose()

    def get_qpos(self):
        return self.qpos

    def get_qvel(self):
        return self.qvel

    def get_root_pose(self):
        return self.global_pose

    def get_pose(self):
        return self.global_pose


class Planner:
    def plan_path(
        self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None
    ):
        return {
            "status": "Success",
            "position": np.array([curr_joint_pos, curr_joint_pos + 0.1]),
            "velocity": np.zeros((2, 3), np.float32),
        }


class Env:
    task_name = "pick_dual_bottles"

    def __init__(self):
        self.take_action_cnt = 0
        self.step_lim = 400
        self.eval_success = False
        self.eval_video_path = None
        self.render_freq = 0
        self.robot = types.SimpleNamespace(
            **{
                side + "_" + kind: cls()
                for side in ["left", "right"]
                for kind, cls in [
                    ("entity", Entity),
                    ("ee", Entity),
                    ("planner", Planner),
                ]
            },
            communication_flag=False,
            get_left_arm_jointState=lambda: [0.0] * 3,
            get_right_arm_jointState=lambda: [0.0] * 3,
        )
        self.bottle1 = Entity()
        self.bottle2 = Entity()
        self.received = []
        self.obs_calls = 0
        self.setup_calls = 0
        self.close_calls = 0
        self.instruction = "fixed instruction"

    def setup_demo(self, *, seed, **kw):
        self.seed = seed
        self.setup_calls += 1
        np.random.seed(seed)

    def play_once(self):
        raise AssertionError("no expert refilter allowed")

    def set_instruction(self, *, instruction):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction

    def get_obs(self):
        self.obs_calls += 1
        return observation()

    def close_env(self, **kw):
        self.close_calls += 1

    def take_action(self, action, *, action_type):
        self.received.append((action.copy(), action_type))
        for side in ["left", "right"]:
            entity = getattr(self.robot, side + "_entity")
            planner = getattr(self.robot, side + "_planner")
            result = planner.plan_path(entity.get_qpos(), Pose(), arms_tag=side)
            entity.qpos = result["position"][-1].copy()
        self.take_action_cnt += 1


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.counter_offset = 0
        self.error = None

    def reset(self):
        self.resets += 1
        return {"type": "reset_ack"}

    def predict_once(self, payload):
        self.calls += 1
        if self.error:
            raise self.error
        return {
            "action": [99.0] * 20,
            "step": self.calls + self.counter_offset,
            "latency_ms": 0.1,
        }

    def close(self):
        pass


class IntegrationChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        overrides = iface._STEP_LIM_OVERRIDES
        iface._STEP_LIM_OVERRIDES = {**overrides, "pick_dual_bottles": 2}
        self.addCleanup(setattr, iface, "_STEP_LIM_OVERRIDES", overrides)

    def setup_trial(self, mode, path="trial", client=None, expected=None):
        capture = make_recording_model(iface)
        payload = capture_payload(iface, capture, observation(), "fixed instruction")
        trace = ContextTrace(
            self.root / path,
            {("clean", 400000): expected or payload_hash(payload)},
            iface._extract_eef_proprio,
            mode,
        )
        client = client or FakeClient()
        model = make_recording_model(iface)
        model._client = client
        proxy = types.SimpleNamespace(get_model=lambda args: model, eval=iface.eval)
        bind_context_interface(proxy, trace, COMMANDS, mode)
        return trace, proxy, proxy.get_model({}), client

    def step(self, trace, proxy, model, env):
        ob = env.get_obs()
        trace.begin(ob, ob, "clean", 400000, env.take_action_cnt)
        proxy.eval(env, model, ob)

    def test_both_modes_trace_real_state_and_identical_saved_targets(self):
        all_actions = []
        for mode in ["resident_idle", "shadow_online"]:
            t, p, m, c = self.setup_trial(mode, mode)
            e = Env()
            iface.reset_model(m)
            for _ in range(2):
                self.step(t, p, m, e)
            rows = [
                json.loads(s)
                for s in (t.out / "seed-400000-trace.jsonl").read_text().splitlines()
            ]
            self.assertEqual(e.obs_calls, 2)
            self.assertEqual(c.resets, 2)
            self.assertEqual(c.calls, 2 if mode == "shadow_online" else 0)
            for i, r in enumerate(rows):
                self.assertEqual(r["fixed_command_index"], i)
                self.assertNotIn("server_step", r)
                self.assertNotIn("server_action", r)
                self.assertEqual(r["env_action"], ROWS[i]["env_action"])
                self.assertEqual(len(r["planner_calls"]), 2)
                self.assertNotEqual(
                    r["before_execution"]["actual_articulations"]["left"]["qpos"],
                    r["after_execution"]["actual_articulations"]["left"]["qpos"],
                )
                self.assertEqual(
                    r["selection"]["online_calls"], 1 if mode == "shadow_online" else 0
                )
                self.assertIn("selection_elapsed_ns", r["selection"])
                self.assertNotIn("roundtrip_ns", r["selection"])
                if mode == "shadow_online":
                    self.assertFalse(r["selection"]["online_action_equals_fixed"])
                    self.assertEqual(r["online_response"]["step"], i + 1)
                else:
                    self.assertIsNone(r["online_response"])
                    self.assertIsNone(r["selection"]["rpc_elapsed_ns"])
            all_actions.append(e.received)
        for a, b in zip(*all_actions):
            np.testing.assert_array_equal(a[0], b[0])

    def test_actual_upstream_one_scene_loop_has_two_resets_observations_and_actions(
        self,
    ):
        for mode in ["resident_idle", "shadow_online"]:
            t, p, m, c = self.setup_trial(mode, mode)
            e = Env()
            node = next(
                n
                for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "eval_policy"
            )
            module = types.ModuleType("context_loop_gate")
            module.np = np
            module.class_decorator = lambda task: e
            module.generate_episode_descriptions = lambda *a: None

            def evaluate(env, model, ob):
                t.begin(ob, ob, "clean", 400000, env.take_action_cnt)
                return p.eval(env, model, ob)

            module.eval_function_decorator = (
                lambda policy, kind: evaluate if kind == "eval" else iface.reset_model
            )
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"),
                module.__dict__,
            )
            install_replay(
                module,
                [
                    {
                        "seed": 400000,
                        "instruction": "fixed instruction",
                        "instruction_choice_length": 10,
                    }
                ],
            )
            args = {
                "task_name": "pick_dual_bottles",
                "policy_name": "stub",
                "render_freq": 0,
                "clear_cache_freq": 5,
                "task_config": "demo_clean",
                "ckpt_setting": "CPU",
            }
            with contextlib.redirect_stdout(io.StringIO()):
                module.eval_policy(
                    "pick_dual_bottles",
                    module.class_decorator("task"),
                    args,
                    m,
                    0,
                    test_num=1,
                    instruction_type="unseen",
                )
            self.assertEqual(
                (
                    e.setup_calls,
                    e.close_calls,
                    e.obs_calls,
                    e.take_action_cnt,
                    c.resets,
                ),
                (1, 1, 2, 2, 2),
            )
            self.assertEqual(e.seed, 400000)
            expected = np.random.RandomState(400000)
            expected.choice(range(10))
            self.assertTrue(
                all(
                    np.array_equal(a, b)
                    for a, b in zip(expected.get_state(), np.random.get_state())
                )
            )

    def test_counter_mismatch_is_saved_then_aborts_before_motion(self):
        client = FakeClient()
        client.counter_offset = 916
        t, p, m, c = self.setup_trial("shadow_online", client=client)
        e = Env()
        with self.assertRaisesRegex(RuntimeError, "counter mismatch"):
            self.step(t, p, m, e)
        failure = json.loads((t.out / "incomplete-trace.json").read_text())
        self.assertEqual(failure["record"]["online_response"]["step"], 917)
        self.assertFalse(failure["record"]["online_counter_valid"])
        self.assertEqual(e.take_action_cnt, 0)
        self.assertEqual(c.calls, 1)

    def test_initial_mismatch_is_saved_without_prediction_or_motion(self):
        t, p, m, c = self.setup_trial("shadow_online", expected="wrong")
        e = Env()
        with self.assertRaisesRegex(RuntimeError, "initial request mismatch"):
            self.step(t, p, m, e)
        self.assertTrue((t.out / "requests/step-0000.json.gz").exists())
        self.assertTrue((t.out / "incomplete-trace.json").exists())
        self.assertEqual((e.take_action_cnt, c.calls), (0, 0))

    def test_timeout_is_saved_without_motion_retry_or_lost_exception(self):
        client = FakeClient()
        client.error = TimeoutError("ambiguous")
        t, p, m, c = self.setup_trial("shadow_online", client=client)
        e = Env()
        with self.assertRaises(TimeoutError) as caught:
            self.step(t, p, m, e)
        self.assertIs(caught.exception, client.error)
        events = [json.loads(s) for s in t.events.read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(e.take_action_cnt, 0)
        self.assertEqual(c.calls, 1)
        with self.assertRaises(RuntimeError):
            m._client.predict({})
        self.assertEqual(c.calls, 1)

    def test_existing_record_directory_is_refused(self):
        self.setup_trial("resident_idle")
        with self.assertRaises(FileExistsError):
            self.setup_trial("resident_idle")

    def test_real_websocket_timeout_does_not_resend_or_continue(self):
        from benchmarks.utils import WSPolicyClient
        from websockets.sync.server import serve

        observed, records, release = [], [], threading.Event()

        def handler(ws):
            for raw in ws:
                message = json.loads(raw)
                observed.append(message["type"])
                if message["type"] == "reset":
                    ws.send(json.dumps({"type": "reset_ack"}))
                else:
                    release.wait(timeout=3)
                    return

        with serve(handler, "127.0.0.1", 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            client = WSPolicyClient(
                f"ws://127.0.0.1:{server.socket.getsockname()[1]}", timeout=0.15
            )
            trial = ContextTrialTransport(
                client, COMMANDS, "shadow_online", records.append
            )
            try:
                trial.reset()
                with self.assertRaises(TimeoutError):
                    trial.predict({"images": {}, "prompt": "CPU timeout", "state": []})
                with self.assertRaises(RuntimeError):
                    trial.predict({})
                self.assertEqual(observed, ["reset", "obs"])
                self.assertEqual(trial.index, 0)
                self.assertEqual(records[-1]["event"], "error")
                self.assertIsNotNone(records[-1]["rpc_elapsed_ns"])
            finally:
                release.set()
                trial.close()
                server.shutdown()
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

    def test_real_loopback_websocket_preserves_rpc_counts_and_response(self):
        from benchmarks.utils import WSPolicyClient
        from websockets.sync.server import serve

        messages = []

        def handler(ws):
            counter = 0
            for raw in ws:
                msg = json.loads(raw)
                messages.append(msg["type"])
                if msg["type"] == "reset":
                    counter = 0
                    reply = {"type": "reset_ack"}
                elif msg["type"] == "ping":
                    reply = {"type": "pong"}
                else:
                    counter += 1
                    reply = {"type": "action", "action": [99.0] * 20, "step": counter}
                ws.send(json.dumps(reply))

        with serve(handler, "127.0.0.1", 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                for mode in ["resident_idle", "shadow_online"]:
                    before = len(messages)
                    records = []
                    client = WSPolicyClient(
                        f"ws://127.0.0.1:{server.socket.getsockname()[1]}", timeout=2
                    )
                    t = ContextTrialTransport(client, COMMANDS, mode, records.append)
                    try:
                        self.assertEqual(t.ping()["type"], "pong")
                        t.reset()
                        t.reset()
                        for i in range(2):
                            self.assertEqual(
                                t.predict({"images": {}, "prompt": "CPU", "state": []})[
                                    "action"
                                ],
                                COMMANDS[i],
                            )
                        self.assertEqual(
                            messages[before:],
                            ["ping", "reset", "reset"]
                            + (["obs", "obs"] if mode == "shadow_online" else []),
                        )
                    finally:
                        t.close()
            finally:
                server.shutdown()
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
