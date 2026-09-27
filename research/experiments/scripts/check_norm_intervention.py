import json,sys
from pathlib import Path
import torch
from robustness_common import Adapter
from norm_common import ARMS,intervene
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/norm-20260921'
sys.path.insert(0,str(ROOT/'OpenWAM'))
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
class Norm(torch.nn.Module):
    _apply_feature_norm=DinoV3VideoEncoder._apply_feature_norm
    def __init__(self):
        super().__init__();self._out_norm=torch.nn.LayerNorm(48,elementwise_affine=False,eps=1e-6)
torch.manual_seed(147);torch.set_num_threads(4);norm=Norm()
z=torch.randn(2,48,3,6,4).to(torch.bfloat16);original=z.clone();adapter=Adapter('mlp').eval()
with torch.no_grad():
 for arm in ARMS:
  y=intervene(z,arm,adapter,norm._apply_feature_norm)
  assert torch.equal(z,original) and torch.equal(y[:,:,1:],z[:,:,1:])
  if arm in ['identity','adapter']:assert torch.equal(z,y)
  else:
   expected=norm._apply_feature_norm(z[:,:,:1]);assert torch.equal(y[:,:,:1],expected)
   assert y[:,:,:1].float().mean(1).abs().max()<.002
   assert (y[:,:,:1].float().square().mean(1).sqrt()-1).abs().max()<.002
 adapter.residual[-1].weight.fill_(.02)
 a=intervene(z,'adapter',adapter,norm._apply_feature_norm)
 n=intervene(z,'adapter_renorm',adapter,norm._apply_feature_norm)
 assert not torch.equal(a,z)
 assert torch.equal(n[:,:,:1],norm._apply_feature_norm(a[:,:,:1]))
result={'all_four_arms_checked':True,'actual_encoder_normalization_method':True,'dtype_shape_and_input_preserved':True,'future_slots_exact':True,'zero_init_identity_exact':True,'post_adapter_order_verified':True}
(OUT/'intervention-cpu-check.json').write_text(json.dumps(result,indent=2));print(result)
