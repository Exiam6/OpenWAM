"""Goal auxiliary with the SAME per-token layer normalization as the frozen readouts."""
import torch
from torch.nn import functional as F


def waypoint_objective(mu, heads, task_index, standardized_target):
    """Use the initial frame only; the caller owns train-only target scaling."""
    if mu.ndim != 5 or mu.shape[2] < 1:
        raise ValueError('Expected B,C,T,H,W posterior means')
    if task_index.shape != (len(mu),) or standardized_target.shape != (len(mu), 2):
        raise ValueError('Task/target and latent batch must align')
    if not torch.isfinite(standardized_target).all():
        raise ValueError('Waypoint targets must be finite')
    if (task_index < 0).any() or (task_index >= len(heads)).any():
        raise ValueError('Invalid task index')
    frame = mu[:, :, 0].float().permute(0,2,3,1)
    frame = F.layer_norm(frame, (mu.shape[1],), eps=1e-6).permute(0,3,1,2)
    x = spatial_pool(frame).flatten(1)
    prediction = torch.stack([heads[int(task)](row) for task, row in zip(task_index, x)])
    return F.mse_loss(prediction.float(), standardized_target.float())


def with_waypoint_objective(native_loss, mu, heads, task_index, target, weight):
    if weight < 0:
        raise ValueError('Negative auxiliary weight')
    if weight == 0:
        # Preserve the native graph and RNG; do not read auxiliary labels/head.
        return native_loss
    return native_loss + weight * waypoint_objective(mu, heads, task_index, target)


def spatial_pool(x):
    """Deterministic 2x2 nonoverlapping block means for even spatial shapes."""
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError('This experiment requires even H,W for fixed 2x2 pooling')
    b,c,h,w=x.shape
    return x.reshape(b,c,2,h//2,2,w//2).float().mean(dim=(3,5)).to(x.dtype)
