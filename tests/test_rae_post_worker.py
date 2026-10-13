import importlib.util
import json
import sys
import time
from pathlib import Path


def load(name):
    path = Path(__file__).resolve().parents[1] / "research/greene/factorial" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_post_queue_collects_before_scoring_once(tmp_path, monkeypatch):
    worker = load("worker")
    monkeypatch.setitem(sys.modules, "worker", worker)
    post = load("post_worker")
    root = tmp_path / "study"
    root.mkdir()
    for name in ["benchmark-env", "policy-env"]:
        path = tmp_path / name / "bin"
        path.mkdir(parents=True)
        (path / "python").symlink_to(sys.executable)
    plan = {
        "deadline": (worker.now() + worker.dt.timedelta(minutes=2)).isoformat(),
        "pause_paths": [],
        "gpus": {"1": "test-uuid"},
        "variants": ["native16", "wide16"],
        "seeds": [42],
        "tasks": ["a", "b", "c"],
        "evaluation_variants": ["native", "wide16"],
        "source_code": str(root),
        "free_polls_required": 1,
        "poll_seconds": 1,
    }
    (root / "protocol.json").write_text(json.dumps(plan))
    (root / "eval-freeze.json").write_text(json.dumps({"files": {}}))
    (root / "claims").mkdir()
    for variant in plan["variants"]:
        (root / "claims" / f"train-{variant}-seed42.json").write_text(json.dumps({"status": "complete"}))
    for name in ["collect_fresh.py", "evaluate_cell.py"]:
        (root / name).write_text("""from pathlib import Path
import json,sys
with (Path(__file__).parent/'order.jsonl').open('a') as f:
 f.write(json.dumps([Path(__file__).name,*sys.argv[1:]])+'\\n')
""")
    (root / "score.py").write_text("pass\n")
    monkeypatch.setattr(post, "ROOT", root)
    monkeypatch.setattr(post, "gpu_status", lambda *args: (True, ["test-uuid", "0", "0", "0", "pci"]))
    monkeypatch.setattr(post.signal, "signal", lambda *args: None)
    sleep = time.sleep
    monkeypatch.setattr(post.time, "sleep", lambda seconds: sleep(0.001))
    post.main("1")
    post.main("1")
    rows = [json.loads(line) for line in (root / "order.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert all(row[0] == "collect_fresh.py" for row in rows[:3])
    assert all(row[0] == "evaluate_cell.py" for row in rows[3:])
    assert json.loads((root / "post-worker-gpu1.json").read_text())["phase"] == "finished"
