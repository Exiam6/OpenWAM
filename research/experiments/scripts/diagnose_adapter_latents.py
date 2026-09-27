#!/usr/bin/env python3
"""Validation-only descriptive diagnostics; never used for model selection."""
import json
from pathlib import Path
import torch
from robustness_common import Adapter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/robustness-20260921'
torch.set_num_threads(4)
data=torch.load(OUT/'adapter-cache.pt',map_location='cpu',weights_only=False)
ck=torch.load(OUT/'adapter-selected.pt',map_location='cpu',weights_only=True)
model=Adapter(ck['kind']).eval();model.load_state_dict(ck['state_dict'])
ids=torch.tensor([i for i,r in enumerate(data['manifest'])if r['split']=='val'])
clean=data['clean'][data['references'][ids]];noisy=data['noisy'][ids]
with torch.no_grad():
 ac=model(clean);an=model(noisy)
rows={}
for name,z in [('clean',clean),('noisy',noisy),('adapted_clean',ac),('adapted_noisy',an)]:
 z=z.float();norm=z.square().mean(1).sqrt()
 rows[name]={'mean_channel_mean':float(z.mean(1).mean()),'mean_token_channel_rms':float(norm.mean()),'std_token_channel_rms':float(norm.std()),'mse_to_clean':float((z-clean.float()).square().mean())}
regions={}
for name,sl in [('head',slice(0,16)),('wrists',slice(16,24))]:
 target=clean[:,:,:,sl].float();x=noisy[:,:,:,sl].float();a=an[:,:,:,sl].float();c=ac[:,:,:,sl].float()
 base=float((x-target).square().mean());restored=float((a-target).square().mean())
 regions[name]={'baseline_noise_mse':base,'adapted_noise_mse':restored,'relative_change_percent':100*(restored/base-1),'clean_distortion_mse':float((c-target).square().mean())}
result={'scope':'descriptive validation diagnostics after architecture selection, before adapted-policy rollouts; no selection or extra fitting','seed':42,'latent_statistics':rows,'layout_regions':regions,'notes':['Head image occupies top 16/24 latent rows; wrists occupy bottom 8/24.','Global DINO/S-VAE mixing means these are token-layout regions, not causally isolated camera features.','These quantities do not determine control sensitivity or success.']}
(OUT/'adapter-latent-diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
