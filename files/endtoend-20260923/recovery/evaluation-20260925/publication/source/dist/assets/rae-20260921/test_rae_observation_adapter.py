import torch
import pytest
from rae_observation_adapter import ObservationAdapter, restoration_loss


def test_initial_identity_and_parameter_count():
    m = ObservationAdapter()
    assert sum(p.numel() for p in m.parameters()) == 9360
    x = torch.randn(2, 48, 3, 4, 4)
    assert torch.equal(m(x), x)


def test_updates_only_observed_frame_without_mutating_input():
    m = ObservationAdapter()
    with torch.no_grad():
        m.residual[-1].bias.fill_(.25)
    x = torch.randn(2, 48, 3, 4, 4)
    old = x.clone()
    y = m(x)
    assert torch.equal(x, old)
    assert torch.allclose(y[:, :, 0], x[:, :, 0] + .25)
    assert torch.equal(y[:, :, 1:], x[:, :, 1:])
    with pytest.raises(ValueError):
        m(torch.empty(2, 48, 0, 4, 4))


def test_loss_does_not_train_teacher_or_encoder():
    m = ObservationAdapter()
    clean = torch.randn(2, 48, 1, 4, 4, requires_grad=True)
    noisy = torch.randn_like(clean, requires_grad=True)
    loss = restoration_loss(m, clean, noisy)
    loss.backward()
    assert clean.grad is None and noisy.grad is None
    assert m.residual[-1].weight.grad.abs().sum() > 0
