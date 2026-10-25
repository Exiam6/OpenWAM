"""Focused checks for the online action-grounding ablation."""

import torch
from types import SimpleNamespace

from openwam.model.video_backbone.encoder.online_action_adapter import (
    CausalActionAdapter,
    action_only_current_latent_gradient,
)
from openwam.model.video_backbone.encoder.dinov2 import DinoV2VideoEncoder
from openwam.model.architectures.base import BaseWAMArchitecture


def test_identity_and_causality():
    torch.manual_seed(17)
    adapter = CausalActionAdapter(4, 3)
    x = torch.randn(2, 4, 3, 2, 2)
    assert torch.equal(adapter(x), x)
    with torch.no_grad():
        adapter.up.weight.normal_(std=0.1)
    altered = x.clone()
    altered[:, :, 1:] += 20
    torch.testing.assert_close(adapter(x)[:, :, :1], adapter(altered)[:, :, :1])


def test_encoder_train_eval_path():
    encoder = DinoV2VideoEncoder(
        torch.nn.Linear(1, 1),
        embed_dim=4,
        patch_size=16,
        num_register_tokens=0,
        online_action_adapter_hidden_dim=3,
    )
    encoder._set_feature_stats(torch.zeros(4), torch.ones(4))
    encoder._batch_encode_pooled_raw = lambda features: features
    features = torch.randn(1, 4, 3, 2, 2)
    encoder.train()
    torch.testing.assert_close(encoder.batch_encode(features), features)
    with torch.no_grad():
        encoder._online_action_adapter.up.weight.normal_(std=0.1)
    encoder.eval()
    torch.testing.assert_close(
        encoder.batch_encode(features), encoder._online_action_adapter(features)
    )


def test_only_action_gradient_reaches_adapter():
    torch.manual_seed(23)
    adapter = CausalActionAdapter(4, 3)
    with torch.no_grad():
        adapter.up.weight.normal_(std=0.1)
    shared = torch.nn.Parameter(torch.tensor(0.7))
    features = torch.randn(2, 4, 3, 2, 2)
    z = adapter(features)
    current = z[:, :, :1]
    future = z.detach()[:, :, 1:]
    world_loss = (shared * current + future.mean()).square().mean()
    action_loss = (shared * current - 1).square().mean()
    expected = torch.autograd.grad(action_loss, tuple(adapter.parameters()), retain_graph=True)
    expected_shared = torch.autograd.grad(world_loss + action_loss, shared, retain_graph=True)[0]
    action_only_current_latent_gradient(action_loss, current)
    (world_loss + action_loss).backward()
    for param, wanted in zip(adapter.parameters(), expected):
        torch.testing.assert_close(param.grad, wanted)
    torch.testing.assert_close(shared.grad, expected_shared)


def test_action_gradient_with_nonreentrant_checkpoint():
    torch.manual_seed(31)
    adapter = CausalActionAdapter(4, 3)
    current = adapter(torch.randn(1, 4, 3, 2, 2))[:, :, :1]
    weight = torch.nn.Parameter(torch.randn(4))

    def trunk(inp):
        return (inp * weight.view(1, 4, 1, 1, 1)).tanh()

    output = torch.utils.checkpoint.checkpoint(trunk, current, use_reentrant=False)
    action_loss = output.square().mean()
    world_loss = (output - 1).square().mean()
    action_only_current_latent_gradient(action_loss, current)
    (action_loss + world_loss).backward()
    assert torch.isfinite(adapter.up.weight.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_joint_loss_does_not_send_world_gradient_to_adapter():
    class Scheduler:
        timesteps = torch.tensor([0.25, 0.75])
        sigmas = torch.tensor([0.25, 0.75])

        @staticmethod
        def add_noise(clean, noise, sigma):
            return (1 - sigma) * clean + sigma * noise

        @staticmethod
        def training_target(clean, noise):
            return noise - clean

    class TinyJointModel(torch.nn.Module):
        compute_loss = BaseWAMArchitecture.compute_loss

        def __init__(self):
            super().__init__()
            self.video_backbone = torch.nn.Module()
            self.video_backbone.video_encoder = torch.nn.Module()
            self.video_backbone.video_encoder._online_action_adapter = CausalActionAdapter(4, 3)
            self.video_backbone.scheduler = Scheduler()
            self.action_backbone = SimpleNamespace(scheduler=Scheduler())
            self.shared = torch.nn.Parameter(torch.tensor(0.7))
            self.dtype = torch.float32
            self.device = torch.device("cpu")

        def forward(self, noisy_actions, action_timestep, *, latents, **_kwargs):
            current = latents[:, :, :1].mean(dim=(1, 2, 3, 4)).view(-1, 1, 1)
            return self.shared * latents, self.shared * noisy_actions + current

        @staticmethod
        def _compute_video_loss(pred, target, *_args):
            return (pred - target).square().mean()

        @staticmethod
        def _compute_action_loss(pred, target, *_args):
            return (pred - target).square().mean()

    torch.manual_seed(37)
    seed_model = TinyJointModel()
    initial = {key: value.clone() for key, value in seed_model.state_dict().items()}
    features = torch.randn(2, 4, 3, 2, 2)
    actions = torch.randn(2, 5, 2)
    results = []
    for world_weight in (0.0, 1.0):
        model = TinyJointModel()
        model.load_state_dict(initial)
        model.train()
        torch.manual_seed(41)
        outcome = model.compute_loss(
            input_latents=features,
            first_frame_latents=features[:, :, :1],
            actions=actions,
            lambda_video=world_weight,
            lambda_action=1.0,
        )
        outcome["loss"].backward()
        adapter_grads = [
            param.grad.clone() for param in model.video_backbone.video_encoder._online_action_adapter.parameters()
        ]
        results.append((adapter_grads, model.shared.grad.clone()))
    for grad_action_only, grad_joint in zip(results[0][0], results[1][0]):
        torch.testing.assert_close(grad_joint, grad_action_only)
    assert not torch.allclose(results[0][1], results[1][1])


if __name__ == "__main__":
    for check in (
        test_identity_and_causality,
        test_encoder_train_eval_path,
        test_only_action_gradient_reaches_adapter,
        test_action_gradient_with_nonreentrant_checkpoint,
        test_joint_loss_does_not_send_world_gradient_to_adapter,
    ):
        check()
        print(f"{check.__name__}: PASS", flush=True)
