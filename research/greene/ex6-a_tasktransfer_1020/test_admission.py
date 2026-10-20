import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_task


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "code.py"
        self.code.write_text("frozen\n")
        self.checkpoint = self.root / "model"
        self.checkpoint.write_bytes(b"model")
        self.p = {"tasks": {"click_bell": {}},
                  "deadline": (dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1)).isoformat(),
                  "code_sha256": {str(self.code): run_task.sha(self.code)},
                  "arms": [{"checkpoint": str(self.checkpoint)}]}
        self.save()
        self.root_patch = patch.object(run_task, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.disk_patch = patch.object(run_task.shutil, "disk_usage",
                                       return_value=type("Disk", (), {"free": 200*(1 << 30)})())
        self.disk_patch.start()
        self.addCleanup(self.disk_patch.stop)

    def save(self):
        (self.root / "protocol.json").write_text(json.dumps(self.p))

    def test_valid(self):
        self.assertEqual(run_task.admission("click_bell"), self.p)

    def test_no_retry_after_partial(self):
        (self.root / "tasks/click_bell").mkdir(parents=True)
        with self.assertRaisesRegex(AssertionError, "no retries"):
            run_task.admission("click_bell")

    def test_tampering(self):
        self.code.write_text("modified")
        with self.assertRaises(AssertionError):
            run_task.admission("click_bell")

    def test_pause(self):
        (self.root / "paused.json").write_text("{}")
        with self.assertRaises(AssertionError):
            run_task.admission("click_bell")

    def test_expired(self):
        self.p["deadline"] = (dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=1)).isoformat()
        self.save()
        with self.assertRaises(AssertionError):
            run_task.admission("click_bell")


if __name__ == "__main__":
    unittest.main()
