from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_trace_preserves_returns_errors_and_restores_methods():
    path = Path(__file__).resolve().parents[1] / "research/greene/factorial/execution_trace.py"
    spec = spec_from_file_location("execution_trace_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    class Scene:
        def step(self):
            return 37

    result = {"status": "Success", "position": [[1], [2]]}

    def right(action):
        raise ValueError("planner error")

    def left(action):
        return result

    robot = SimpleNamespace(left_plan_path=left, right_plan_path=right)
    scene = Scene()
    other = Scene()
    original = Scene.step
    env = SimpleNamespace(robot=robot, scene=scene)
    with module.track_execution(env, Scene) as record:
        assert robot.left_plan_path([1]) is result
        with pytest.raises(ValueError, match="planner error"):
            robot.right_plan_path([2])
        assert scene.step() == other.step() == 37
        assert record["physics_ticks"] == 1
        assert record["planner_calls"][0] == {"side": "left", "status": "Success", "path_points": 2}
        assert "planner error" in record["planner_calls"][1]["exception"]
    assert Scene.step is original
    assert robot.left_plan_path is left and robot.right_plan_path is right
