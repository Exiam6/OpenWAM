"""Fixed diagnostic on the existing validation cache; no model selection."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from robustness_common import Adapter
from norm_common import ARMS,intervene
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/norm-20260921';OLD=ROOT/'results/robustness-20260921'
sys.path.insert(0,str(ROOT/'OpenWAM'))
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
class Norm(torch.nn.Module):
    _apply_feature_norm=DinoV3VideoEncoder._apply_feature_norm
    def __init__(self):
        super().__init__();self._out_norm=torch.nn.LayerNorm(48,elementwise_affine=False,eps=1e-6)
norm=Norm();torch.set_num_threads(4)
data=torch.load(OLD/'adapter-cache.pt',map_location='cpu',weights_only=False)
ck=torch.load(OLD/'adapter-selected.pt',map_location='cpu',weights_only=True)
model=Adapter(ck['kind']).eval().requires_grad_(False);model.load_state_dict(ck['state_dict'],strict=True)
ids=torch.tensor([i for i,r in enumerate(data['manifest']) if r['split']=='val'])
clean=data['clean'][data['references'][ids]];noisy=data['noisy'][ids]
episodes=np.array([data['manifest'][i]['episode'] for i in ids.tolist()]);assert len(np.unique(episodes))==5
results={}
with torch.inference_mode():
    for arm in ARMS:
        c=intervene(clean,arm,model,norm._apply_feature_norm);n=intervene(noisy,arm,model,norm._apply_feature_norm)
        mse=(n.float()-clean.float()).square().flatten(1).mean(1)
        distortion=(c.float()-clean.float()).square().flatten(1).mean(1)
        stats={}
        for label,z in [('clean',c),('noise',n)]:
            z=z.float();rms=z.square().mean(1).sqrt()
            stats[label]={'mean_token_rms':float(rms.mean()),'std_token_rms':float(rms.std()),'mean_abs_token_mean':float(z.mean(1).abs().mean())}
        results[arm]={'noise_restoration_mse':float(mse.mean()),'clean_distortion_mse':float(distortion.mean()),'max_clean_change':float((c-clean).abs().max()),'statistics':stats,'episodes':[{'episode':int(e),'noise_restoration_mse':float(mse[episodes==e].mean()),'clean_distortion_mse':float(distortion[episodes==e].mean())} for e in np.unique(episodes)]}
    rng=np.random.default_rng(1941);boot=rng.integers(0,5,(10000,5));paired={}
    for a,b in [('adapter','adapter_renorm'),('identity','renorm')]:
        paired[b+'_minus_'+a]={}
        for metric in ['noise_restoration_mse','clean_distortion_mse']:
            delta=np.array([y[metric]-x[metric] for x,y in zip(results[a]['episodes'],results[b]['episodes'])])
            paired[b+'_minus_'+a][metric]={'mean':float(delta.mean()),'episode_bootstrap_95ci':np.quantile(delta[boot].mean(1),[.025,.975]).tolist()}
result={'scope':'existing five-episode validation cache; diagnostic only, no selection, no held-out demonstration test opened','arms':results,'paired_differences':paired}
(OUT/'offline.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
