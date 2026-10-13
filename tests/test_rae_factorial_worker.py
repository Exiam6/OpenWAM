"""Exercise real CPU child processes; no GPU, checkpoint, or network."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "research/greene/factorial/worker.py"
    spec = importlib.util.spec_from_file_location("factorial_worker_test", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


@pytest.mark.parametrize("fail_preflight", [False, True])
def test_queue_preflight_gate_and_exactly_once(tmp_path, monkeypatch, fail_preflight):
    worker = module()
    root = tmp_path / "study"
    root.mkdir()
    binpath = tmp_path / "policy-env/bin"
    binpath.mkdir(parents=True)
    (binpath / "python").symlink_to(sys.executable)
    p = {
        "deadline": (worker.now() + worker.dt.timedelta(minutes=2)).isoformat(),
        "gpus": {"1": "test-uuid"},
        "variants": ["native16", "wide16"],
        "seeds": [42, 43, 44],
        "pause_paths": [],
        "free_polls_required": 1,
        "poll_seconds": 1,
        "source_code": str(root),
    }
    (root / "protocol.json").write_text(json.dumps(p))
    (root / "code-freeze.json").write_text(json.dumps({"files": {}}))
    (root / "train.py").write_text(
        """import json,sys
from pathlib import Path
r=Path(__file__).parent
variant,seed=sys.argv[1:3]
preflight='--preflight' in sys.argv
with (r/'execution-order.jsonl').open('a') as f:
 f.write(json.dumps([variant,seed,preflight])+'\\n')
if preflight and variant=='wide16' and FAIL:
 sys.exit(7)
d=r/('preflight' if preflight else 'runs')/f'{variant}-seed{seed}'
d.mkdir(parents=True)
(d/'result.json').write_text(json.dumps({'complete':True}))
""".replace("FAIL", repr(fail_preflight))
    )
    monkeypatch.setattr(worker, "ROOT", root)
    monkeypatch.setattr(worker, "gpu_status", lambda *args: (True, ["test-uuid", "0", "0", "0", "pci"]))
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    sleep = time.sleep
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: sleep(0.001))
    if fail_preflight:
        with pytest.raises(RuntimeError, match="preflight failed"):
            worker.run("1")
    else:
        worker.run("1")
        worker.run("1")  # Terminal attempts cannot run again.
    records = [json.loads(line) for line in (root / "execution-order.jsonl").read_text().splitlines()]
    assert len(records) == (2 if fail_preflight else 8)
    assert all(row[2] for row in records[:2])
    assert all(not row[2] for row in records[2:])
    assert len({tuple(row) for row in records}) == len(records)
    state = json.loads((root / "worker-gpu1.json").read_text())
    assert state["phase"] == ("stopped" if fail_preflight else "training_queue_finished")


@pytest.mark.parametrize(
    "stamp,allowed",
    [
        ("2026-10-12T05:59:59+00:00", False),  # Chicago 00:59
        ("2026-10-12T06:00:00+00:00", True),
        ("2026-10-12T13:59:59+00:00", True),
        ("2026-10-12T14:00:00+00:00", False),
        ("2026-11-01T06:30:00+00:00", True),  # Both repeated 01:30 hours
        ("2026-11-01T07:30:00+00:00", True),
        ("2026-11-01T15:00:00+00:00", False),
        ("2027-03-14T07:00:00+00:00", True),
        ("2027-03-14T14:00:00+00:00", False),
    ],
)
def test_nightly_resource_boundaries(stamp, allowed):
    worker = module()
    policy = {
        "gpus": {"0": "extra", "1": "baseline"},
        "baseline_gpus": ["1"],
        "timezone": "America/Chicago",
        "start_hour": 1,
        "end_hour": 9,
    }
    current = worker.dt.datetime.fromisoformat(stamp)
    deadline = current + worker.dt.timedelta(days=1)
    end = worker.resource_end(policy, "0", deadline, current)
    assert (end is not None) == allowed
    if end is not None:
        assert end.astimezone(worker.ZoneInfo("America/Chicago")).hour == 9
    assert worker.resource_end(policy, "1", deadline, current) == deadline
    assert worker.resource_end(policy, "7", deadline, current) is None
    assert worker.resource_end(policy, "1", current, current) is None


def test_nightly_admission_reserves_full_attempt_and_cleanup():
    worker = module()
    policy = {
        "gpus": {"0": "extra", "1": "baseline"},
        "baseline_gpus": ["1"],
        "timezone": "America/Chicago",
        "start_hour": 1,
        "end_hour": 9,
    }
    deadline = worker.dt.datetime.fromisoformat("2026-10-12T11:52:56+00:00")
    current = deadline - worker.dt.timedelta(seconds=10920)
    assert worker.admit(policy, "0", deadline, current, "train") == deadline - worker.dt.timedelta(seconds=60)
    assert worker.admit(policy, "0", deadline, current + worker.dt.timedelta(seconds=1), "train") is None
    # Evaluation cannot fit in this remaining interval, collection can.
    assert worker.admit(policy, "0", deadline, current, "evaluate") is None
    assert worker.admit(policy, "0", deadline, current, "collect") is not None
    assert worker.admit(policy, "1", deadline, current, "evaluate") == deadline


def test_extra_gpu_does_not_query_or_claim_outside_window(tmp_path, monkeypatch):
    worker = module()
    current = worker.dt.datetime.fromisoformat("2026-10-11T18:00:00+00:00")
    protocol = {
        "deadline": (current + worker.dt.timedelta(days=1)).isoformat(),
        "gpus": {"1": "baseline"},
        "variants": ["native"],
        "seeds": [42],
        "pause_paths": [],
    }
    policy = {
        "gpus": {"0": "extra", "1": "baseline"},
        "baseline_gpus": ["1"],
        "timezone": "America/Chicago",
        "start_hour": 1,
        "end_hour": 9,
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "resource-policy.json").write_text(json.dumps(policy))
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "now", lambda: current)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "gpu_status", lambda *args: pytest.fail("unauthorized GPU query"))

    def stop(seconds):
        assert json.loads((tmp_path / "worker-gpu0.json").read_text())["phase"] == "waiting_for_authorized_window"
        raise RuntimeError("test stop")

    monkeypatch.setattr(worker.time, "sleep", stop)
    with pytest.raises(RuntimeError, match="test stop"):
        worker.run("0")
    assert not list((tmp_path / "claims").glob("*.json"))
