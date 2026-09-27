"""Frozen inference interventions; no trainable weights added."""
import torch
ARMS = ('identity', 'renorm', 'adapter', 'adapter_renorm')

def intervene(z, arm, adapter, normalize):
    if arm not in ARMS:
        raise ValueError(arm)
    output = adapter(z) if arm.startswith('adapter') else z
    if arm.endswith('renorm'):
        first = normalize(output[:, :, :1])
        output = torch.cat((first, output[:, :, 1:]), dim=2)
    assert output.shape == z.shape and output.dtype == z.dtype
    assert torch.equal(output[:, :, 1:], z[:, :, 1:])
    assert torch.isfinite(output).all()
    return output
