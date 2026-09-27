"""CPU real-child supervisor and assembled client gates; no model/simulator."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from context_runner import run_owned_trial


class LifecycleChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server = self.root / "server.py"
        self.client = self.root / "client.py"
        self.server.write_text("""import json,os,sys,time,subprocess
from pathlib import Path
out=Path(sys.argv[1]);mode=sys.argv[2]
if mode=='early':raise SystemExit(7)
if mode=='hang':time.sleep(30)
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
(out/'owned-child.pid').write_text(str(child.pid))
(out/'server-ready.json').write_text(json.dumps({'model_initialized':True}))
time.sleep(30)
""")
        self.client.write_text("""import sys,time
from pathlib import Path
out=Path(sys.argv[1]);mode=sys.argv[2]
(out/'client-entered').write_text('yes')
if mode=='hang':time.sleep(30)
raise SystemExit(7 if mode=='fail' else 0)
""")

    def run_case(self, server_mode="ready", client_mode="ok", budget=2, startup=0.6):
        out = self.root / "trial"
        return out, run_owned_trial(
            out,
            [sys.executable, str(self.server), str(out), server_mode],
            [sys.executable, str(self.client), str(out), client_mode],
            os.environ.copy(),
            self.root,
            time.monotonic() + budget,
            startup_limit=startup,
            poll=0.02,
        )

    def test_success_cleans_owned_descendant_and_preserves_unrelated_process(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
        )
        try:
            out, result = self.run_case()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["client_exit_code"], 0)
            self.assertIsNone(unrelated.poll())
            child = int((out / "owned-child.pid").read_text())
            stat = Path(f"/proc/{child}/stat")
            if stat.exists():
                self.assertEqual(stat.read_text().rsplit(")", 1)[1].split()[0], "Z")
            self.assertTrue((out / "trial-status.json").exists())
            with self.assertRaises(FileExistsError):
                self.run_case()
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=2)

    def test_server_early_exit_never_starts_client(self):
        out, result = self.run_case("early")
        self.assertEqual(result["status"], "failed")
        self.assertIn("before ready", result["error"])
        self.assertFalse((out / "client-entered").exists())

    def test_client_nonzero_is_retained_and_stops_server(self):
        out, result = self.run_case(client_mode="fail")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["client_exit_code"], 7)
        self.assertTrue(result["cleanup"])

    def test_startup_timeout_is_bounded(self):
        out, result = self.run_case("hang", budget=0.5, startup=0.2)
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertLess(result["elapsed_seconds"], 2.5)
        self.assertFalse((out / "client-entered").exists())

    def test_client_timeout_is_bounded_and_retained(self):
        out, result = self.run_case(client_mode="hang", budget=0.5)
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertLess(result["elapsed_seconds"], 2.5)
        self.assertTrue((out / "client-entered").exists())

    def test_pilot_cli_keeps_abba_order_and_stops_after_first_failure(self):
        import context_runner

        for fail_second in (False, True):
            root = self.root / str(fail_second)
            root.mkdir()
            study = root / "study"
            study.mkdir()
            trials = [
                {"id": name, "mode": mode}
                for name, mode in [
                    ("A0", "resident_idle"),
                    ("B0", "shadow_online"),
                    ("B1", "shadow_online"),
                    ("A1", "resident_idle"),
                ]
            ]
            (study / "run-plan.json").write_text(
                json.dumps({"wall_limit_seconds": 600, "gpu": "mock", "trials": trials})
            )
            (study / "source-manifest.json").write_text(
                json.dumps({"runtime_files": {}, "study_files": {}})
            )
            observed = []

            def fake_trial(out, server, client, env, cwd, deadline):
                observed.append((out.name, env["CONTEXT_MODE"]))
                self.assertIn("context_client.py", " ".join(client))
                self.assertEqual(env["ROBOTWIN_TEST_NUM"], "1")
                return {
                    "status": "failed"
                    if fail_second and len(observed) == 2
                    else "completed"
                }

            with (
                patch.object(
                    sys, "argv", ["runner", "--root", str(root), "--study", str(study)]
                ),
                patch.object(context_runner, "preflight"),
                patch.object(context_runner, "run_owned_trial", fake_trial),
                patch.object(context_runner.time, "sleep"),
                patch.object(context_runner.signal, "signal"),
            ):
                with self.assertRaises(SystemExit) as stopped:
                    context_runner.main()
            self.assertEqual(stopped.exception.code, 1 if fail_second else 0)
            expected = [(t["id"], t["mode"]) for t in trials[: 2 if fail_second else 4]]
            self.assertEqual(observed, expected)
            self.assertEqual(
                len(json.loads((study / "launch.json").read_text())["trials"]),
                len(expected),
            )

    def test_assembled_native_client_records_two_fixed_actions_and_closes(self):
        import context_client
        import norm_client
        from capture_repeat_inputs import capture_payload, make_recording_model
        from check_context_integration import (
            COMMANDS,
            Env,
            FakeClient,
            iface,
            observation,
        )
        from repeat_common import payload_hash

        for mode in ["resident_idle", "shadow_online"]:
            study = self.root / mode
            study.mkdir()
            out = study / "trial"
            payload = capture_payload(
                iface, make_recording_model(iface), observation(), "fixed instruction"
            )
            (study / "scenes.json").write_text(
                json.dumps({"scenes": [{"seed": 400000}]})
            )
            (study / "commands.json").write_text(
                json.dumps(
                    {
                        "commands": [{"server_action": a} for a in COMMANDS],
                        "initial_request_sha256": payload_hash(payload),
                    }
                )
            )
            client = FakeClient()
            client.closed = False
            client.close = lambda: setattr(client, "closed", True)
            model = make_recording_model(iface)
            model._client = client

            def original_main():
                m = iface.get_model({})
                e = Env()
                iface.reset_model(m)
                for i in range(2):
                    ob = e.get_obs()
                    altered = norm_client.client.change_observation(
                        ob, "clean", 400000, i
                    )
                    iface.eval(e, m, altered)
                (out / "summary.json").write_text(
                    json.dumps(
                        {
                            "records": [
                                {
                                    "seed": 400000,
                                    "actions": e.take_action_cnt,
                                    "calls": e.obs_calls,
                                }
                            ]
                        }
                    )
                )

            env = {
                "CONTEXT_MODE": mode,
                "CONTEXT_STUDY": str(study),
                "ROBUST_RUN_DIR": str(out),
                "ROBOTWIN_TEST_NUM": "1",
                "NORM_SIGMA": "0",
            }
            with (
                patch.dict(os.environ, env),
                patch.dict(sys.modules, {"openwam2robotwin_interface": iface}),
                patch.object(iface, "_STEP_LIM_OVERRIDES", {}),
                patch.object(iface, "get_model", lambda args: model),
                patch.object(iface, "eval", iface.eval),
                patch.object(norm_client.client, "main", original_main),
                patch.object(norm_client.client, "change_observation", lambda *a: a[0]),
                patch.object(norm_client, "main", lambda: norm_client.client.main()),
            ):
                context_client.main()
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["local_commands"], 2)
            self.assertEqual(summary["reset_count"], 2)
            self.assertTrue(client.closed)
            self.assertEqual(client.calls, 2 if mode == "shadow_online" else 0)


if __name__ == "__main__":
    unittest.main()
