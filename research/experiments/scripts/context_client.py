"""Native one-scene, two-command context client; launched only by frozen runner."""

import json
import os
from pathlib import Path

import norm_client
from context_trace import ContextTrace, bind_context_interface

ROOT = Path(__file__).resolve().parents[1]


def main():
    mode = os.environ["CONTEXT_MODE"]
    study = Path(os.environ["CONTEXT_STUDY"])
    out = Path(os.environ["ROBUST_RUN_DIR"])
    scenes = json.loads((study / "scenes.json").read_text())["scenes"]
    assert (
        len(scenes) == 1
        and scenes[0]["seed"] == 400000
        and os.environ["ROBOTWIN_TEST_NUM"] == "1"
    )
    fixed = json.loads((study / "commands.json").read_text())
    commands = [r["server_action"] for r in fixed["commands"]]
    expected = fixed["initial_request_sha256"]
    original_main = norm_client.client.main
    models = []

    def wrapped_main():
        import openwam2robotwin_interface as interface

        assert "pick_dual_bottles" not in interface._STEP_LIM_OVERRIDES
        interface._STEP_LIM_OVERRIDES = {
            **interface._STEP_LIM_OVERRIDES,
            "pick_dual_bottles": 2,
        }
        trace = ContextTrace(
            out, {("clean", 400000): expected}, interface._extract_eef_proprio, mode
        )
        bind_context_interface(interface, trace, commands, mode)
        get = interface.get_model

        def remember(args):
            model = get(args)
            models.append(model)
            return model

        interface.get_model = remember
        change = norm_client.client.change_observation

        def traced_change(raw, condition, seed, step):
            # Save every initial payload in ContextTrace before its mismatch gate.
            # Clean has no observation intervention; norm's old image-only gate
            # would raise before the full request could be saved.
            assert condition == "clean" and os.environ["NORM_SIGMA"] == "0"
            trace.begin(raw, raw, condition, seed, step)
            return raw

        norm_client.client.change_observation = traced_change
        try:
            original_main()
            assert trace.current is None
            summary = json.loads((out / "summary.json").read_text())
            assert (
                len(summary["records"]) == 1
                and summary["records"][0]["actions"]
                == summary["records"][0]["calls"]
                == 2
            )
            assert (
                len(models) == 1
                and models[0]._client.index == 2
                and models[0]._client.resets == 2
            )
            summary.update(
                role="two-command context diagnostic; not task-success evaluation",
                mode=mode,
                local_commands=2,
                online_obs_calls=2 if mode == "shadow_online" else 0,
                reset_count=models[0]._client.resets,
            )
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        except BaseException as error:
            if (
                trace.current is not None
                and not (out / "incomplete-trace.json").exists()
            ):
                trace.abort(error)
            raise
        finally:
            norm_client.client.change_observation = change
            for model in models:
                model._client.close()

    norm_client.client.main = wrapped_main
    norm_client.main()


if __name__ == "__main__":
    main()
