import importlib.util
from pathlib import Path

import pytest


def score_module():
    file = Path(__file__).resolve().parents[1] / "research/greene/factorial/score.py"
    spec = importlib.util.spec_from_file_location("rae_score_test", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_data():
    plan = {
        "evaluation_variants": ["raw768_native", "raw768_wide_shift16"],
        "seeds": [42, 43, 44],
        "tasks": ["task_a", "task_b", "task_c"],
    }
    scenes = {task: [{"seed": i} for i in range(20)] for task in plan["tasks"]}
    records = [
        {
            "route": v,
            "training_seed": s,
            "task": t,
            "scene_seed": i,
            "status": "complete",
            "success": v == "raw768_wide_shift16",
        }
        for v in plan["evaluation_variants"]
        for s in plan["seeds"]
        for t in plan["tasks"]
        for i in range(20)
    ]
    return plan, scenes, records


def test_missing_results_prevent_success_claim():
    plan, scenes, records = fixture_data()
    result = score_module().summarize(plan, scenes, records[:-1])
    assert not result["complete"] and not result["verified_success_rate_improvement"]
    assert result["attempted"] == 359 and result["planned"] == 360


def test_complete_positive_and_null_results():
    plan, scenes, records = fixture_data()
    score = score_module()
    result = score.summarize(plan, scenes, records, draws=100)
    assert result["verified_success_rate_improvement"]
    assert result["primary"]["scene_cluster_paired_95pct_interval"] == [1.0, 1.0]
    for row in records:
        row["success"] = False
    result = score.summarize(plan, scenes, records, draws=100)
    assert not result["verified_success_rate_improvement"]
    assert result["primary"]["task_balanced_success_difference"] == 0


def test_technical_failure_is_not_success_and_duplicates_rejected():
    plan, scenes, records = fixture_data()
    for row in records:
        if row["route"] == "raw768_wide_shift16":
            row["status"] = "infrastructure_failure"
    score = score_module()
    result = score.summarize(plan, scenes, records, draws=100)
    assert not result["verified_success_rate_improvement"]
    assert result["counts"]["raw768_wide_shift16"]["successes"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        score.summarize(plan, scenes, records + [records[0]])
