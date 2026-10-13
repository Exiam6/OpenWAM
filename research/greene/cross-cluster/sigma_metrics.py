"""Comparable video-noise metrics without changing training loss semantics."""

import math
from contextlib import contextmanager

import torch


@contextmanager
def fixed_video_sigma(scheduler, sigma):
    """Temporarily evaluate at one exact sigma with unit video-MSE weights.

    Replacing (rather than mutating) buffers preserves the original tensors,
    including after exceptions. Keep the buffer length so native RNG calls do
    not change. The action scheduler is deliberately untouched. Float32 sigma
    values are subsequently rounded to the model dtype by native compute_loss.
    This is for single-threaded evaluation only, never training or serving.
    """
    if not math.isfinite(sigma) or not 0 <= sigma <= 1:
        raise ValueError("sigma must be finite and in [0, 1]")
    names = ("sigmas", "timesteps", "linear_timesteps_weights")
    saved = {name: getattr(scheduler, name) for name in names}
    if any(not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.numel() for value in saved.values()):
        raise ValueError("scheduler needs nonempty one-dimensional training buffers")
    if len({len(value) for value in saved.values()}) != 1:
        raise ValueError("scheduler buffer lengths must match")
    try:
        scheduler.sigmas = torch.full_like(saved["sigmas"], sigma)
        scheduler.timesteps = torch.full_like(saved["timesteps"], sigma * scheduler.num_train_timesteps)
        scheduler.linear_timesteps_weights = torch.ones_like(saved["linear_timesteps_weights"])
        yield
    finally:
        for name, value in saved.items():
            setattr(scheduler, name, value)


def recover_unweighted_curve(curve, scheduler):
    """Recover MSE from old single-index curves; refuse incompatible records.

    Native loss is mean(per-sample MSE * weight). Division of a scalar average
    is valid here only because every sample used the same fixed timestep index.
    This does not remove the old nearest-grid sigma mismatch or provide CIs.
    """
    n = len(scheduler.timesteps)
    rows = []
    for row in curve:
        index = row["index"]
        if not isinstance(index, int) or not 0 <= index < n:
            raise ValueError("invalid timestep index")
        if (int(index / n * n), int((index + 1) / n * n)) != (index, index + 1):
            raise ValueError("old floating-point boundaries did not pin one index")
        actual = float(scheduler.sigmas[index])
        if not math.isclose(actual, row["actual_sigma"], rel_tol=0, abs_tol=1e-7):
            raise ValueError("recorded sigma does not match reconstructed scheduler")
        weight = float(scheduler.linear_timesteps_weights[index])
        loss = row["loss_video"]
        if not math.isfinite(weight) or weight <= 0 or not math.isfinite(loss) or loss < 0:
            raise ValueError("need finite nonnegative loss and strictly positive weight")
        rows.append({**row, "training_weight": weight, "video_mse": loss / weight})
    return rows
