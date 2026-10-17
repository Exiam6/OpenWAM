"""Observe native planner returns and physics ticks without changing actions."""

from contextlib import contextmanager


@contextmanager
def track_execution(env, scene_type):
    events = []
    state = {"physics_ticks": 0, "planner_calls": events}
    original_step = scene_type.step
    robot = env.robot
    originals = {side: getattr(robot, side + "_plan_path") for side in ["left", "right"]}

    def step(scene, *args, **kwargs):
        value = original_step(scene, *args, **kwargs)
        if scene is env.scene:
            state["physics_ticks"] += 1
        return value

    def wrap(side, original):
        def plan(*args, **kwargs):
            try:
                value = original(*args, **kwargs)
            except Exception as exc:
                events.append({"side": side, "exception": repr(exc)})
                raise
            events.append(
                {
                    "side": side,
                    "status": str(value.get("status")),
                    "path_points": len(value["position"]) if "position" in value else None,
                }
            )
            return value

        return plan

    try:
        scene_type.step = step
        for side, original in originals.items():
            setattr(robot, side + "_plan_path", wrap(side, original))
        yield state
    finally:
        scene_type.step = original_step
        for side, original in originals.items():
            setattr(robot, side + "_plan_path", original)
