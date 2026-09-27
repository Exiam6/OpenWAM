"""Contract checks independent of pretrained weights and held-out data."""
import json, torch
from openwam.model.video_backbone.encoder.dinov3_temporal import ConvexTemporalPool
from openwam.model.video_backbone.encoder.dinov3 import _causal_temporal_pool

def check():
    torch.manual_seed(426)
    x=torch.randn(2,8,9,3,2);p=ConvexTemporalPool(8)
    for dtype in (torch.float32,torch.bfloat16):
        a=x.to(dtype);assert torch.equal(p(a),_causal_temporal_pool(a))
    with torch.no_grad():p.logits.normal_()
    y=p(x);assert torch.equal(y[:,:,:1],x[:,:,:1])
    changed=x.clone();changed[:,:,5:]+=100
    assert torch.equal(y[:,:,:2],p(changed)[:,:,:2])
    assert torch.equal(p(x[:,:,:1]),x[:,:,:1])
    y.square().mean().backward();assert p.logits.grad.isfinite().all() and p.logits.grad.abs().sum()>0
    w=p.weights();assert torch.allclose(w.sum(-1),torch.ones(8)) and w.min()>=.125 and w.max()<=.625
    q=ConvexTemporalPool(8);q.load_state_dict(p.state_dict(),strict=True);assert torch.equal(q(x),p(x))
    for t in [0,2,4,8]:
        try:p(torch.zeros(2,8,t,3,2))
        except ValueError:pass
        else:raise AssertionError(t)
    return {'passed':True,'checks':['fp32_bf16_exact_mean_init','first_frame_identity','no_later_group_leakage','single_frame_identity','nonzero_finite_gradient','convex_bound','strict_reload','invalid_length_rejected']}
if __name__=='__main__':print(json.dumps(check()))
