"""CPU regressions for cross-shift metric comparability; no model checkpoint."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler

SCRIPTS = Path(__file__).resolve().parents[1] / "research/greene/cross-cluster"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = load_script("sigma_metrics")
sys.modules["sigma_metrics"] = metrics
evaluator = load_script("eval_sigma_curve_v2")


def scheduler(shift):
    result = FlowMatchScheduler("Wan")
    result.set_timesteps(1000, training=True, shift=shift)
    return result


def video_loss(schedule):
    model = SimpleNamespace(video_backbone=SimpleNamespace(scheduler=schedule))
    # A clean prefix and a padded last frame must not contribute to the metric.
    prediction = torch.tensor([100.0, 2.0, 200.0]).reshape(1, 1, 3, 1, 1)
    return BaseWAMArchitecture._compute_video_loss(
        model,
        prediction,
        torch.zeros_like(prediction),
        torch.tensor([500]),
        {"first_frame_latents": prediction[:, :, :1], "video_is_pad": torch.tensor([[False, True]])},
        torch.device("cpu"),
    )


def test_native_weight_confounds_identical_predictions_but_fixed_metric_does_not():
    low, high = scheduler(5), scheduler(16.4317)
    assert float(video_loss(low)) != pytest.approx(float(video_loss(high)))
    for schedule in (low, high):
        with metrics.fixed_video_sigma(schedule, 0.5):
            assert float(video_loss(schedule)) == 4.0
            assert torch.equal(schedule.sigmas, torch.full((1000,), 0.5))
            assert torch.equal(schedule.timesteps, torch.full((1000,), 500.0))


@pytest.mark.parametrize("fail", [False, True])
def test_restore_tensor_identity_even_on_failure(fail):
    schedule = scheduler(5)
    names = ("sigmas", "timesteps", "linear_timesteps_weights")
    original = {name: getattr(schedule, name) for name in names}

    def run():
        with metrics.fixed_video_sigma(schedule, 0.4):
            outer = schedule.sigmas
            with metrics.fixed_video_sigma(schedule, 0.9):
                assert float(schedule.sigmas[0]) == pytest.approx(0.9)
            assert schedule.sigmas is outer
            if fail:
                raise RuntimeError("test error")

    if fail:
        with pytest.raises(RuntimeError, match="test error"):
            run()
    else:
        run()
    assert all(getattr(schedule, name) is value for name, value in original.items())


@pytest.mark.parametrize("sigma", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        with metrics.fixed_video_sigma(scheduler(5), sigma):
            pytest.fail("invalid context entered")


def test_recover_single_index_average():
    schedule = scheduler(16.4317)
    index = 500
    weight = float(schedule.linear_timesteps_weights[index])
    row = {
        "index": index,
        "actual_sigma": float(schedule.sigmas[index]),
        "loss_video": sum(v * weight for v in [1.0, 3.0, 8.0]) / 3,
    }
    assert metrics.recover_unweighted_curve([row], schedule)[0]["video_mse"] == pytest.approx(4.0)
    assert "video_mse" not in row


@pytest.mark.parametrize(
    "bad", [{"index": -1}, {"actual_sigma": 0.0}, {"loss_video": float("nan")}, {"loss_video": -1.0}]
)
def test_recovery_rejects_incompatible_records(bad):
    schedule = scheduler(5)
    row = {"index": 500, "actual_sigma": float(schedule.sigmas[500]), "loss_video": 1.0}
    with pytest.raises(ValueError):
        metrics.recover_unweighted_curve([{**row, **bad}], schedule)


def test_recovery_rejects_zero_weight():
    schedule = scheduler(5)
    assert float(schedule.linear_timesteps_weights[0]) == 0.0
    with pytest.raises(ValueError, match="positive weight"):
        metrics.recover_unweighted_curve([{"index": 0, "actual_sigma": 1.0, "loss_video": 0.0}], schedule)


def test_evaluator_keeps_paired_draws_metadata_and_action_scheduler():
    action = scheduler(5)
    action_weights = action.linear_timesteps_weights
    cache = {"metadata": [{"split": "train"}, {"split": "dev", "episode": 7}]}
    outputs = []
    for shift in [5, 16.4317]:
        video = scheduler(shift)
        original = video.sigmas
        model = SimpleNamespace(
            training=False,
            dtype=torch.bfloat16,
            video_backbone=SimpleNamespace(scheduler=video),
            action_backbone=SimpleNamespace(scheduler=action),
        )

        def inputs_for(cache, ids, variant, checkpoint):
            assert ids == [1] and checkpoint is False
            return {}

        def paired_loss(model, inputs, seed):
            assert inputs == {"min_timestep_boundary": 0.0, "max_timestep_boundary": 1.0}
            assert action.linear_timesteps_weights is action_weights
            return {"loss_video": video_loss(video), "loss_action": seed / 1e7}

        outputs.append(evaluator.evaluate_curve(model, cache, "raw", inputs_for, paired_loss, targets=[0.1]))
        assert video.sigmas is original
    assert outputs[0] == outputs[1]
    row = outputs[0][0]
    assert row["video_mse"] == 4.0
    assert row["model_sigma"] == float(torch.tensor(0.1, dtype=torch.bfloat16))
    assert [v["noise_seed"] for v in row["values"]] == [7000010, 7000011, 7000012]
    assert all(v["metadata"]["episode"] == 7 for v in row["values"])


def test_guard_stops_evaluation_and_restores_scheduler():
    video, action = scheduler(5), scheduler(5)
    original = video.sigmas
    model = SimpleNamespace(
        training=False,
        dtype=torch.bfloat16,
        video_backbone=SimpleNamespace(scheduler=video),
        action_backbone=SimpleNamespace(scheduler=action),
    )
    calls = []

    def guard():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("study paused")

    def never_called(*args, **kwargs):
        pytest.fail("model evaluation must not run after pause")

    with pytest.raises(RuntimeError, match="study paused"):
        evaluator.evaluate_curve(
            model,
            {"metadata": [{"split": "dev"}]},
            "raw",
            never_called,
            never_called,
            targets=[0.1],
            guard=guard,
        )
    assert video.sigmas is original
