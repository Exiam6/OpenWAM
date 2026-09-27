import argparse
import json
import logging
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file
from robustness_common import Adapter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/robustness-20260921'
sys.path.insert(0, str(ROOT/'OpenWAM'))
from openwam.deploy.server import build_server_from_config

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adapter');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO);torch.manual_seed(42);torch.set_num_threads(4)
    cfg=OmegaConf.load(ROOT/'OpenWAM/configs/deploy.yaml');cfg.optimization.compile.enabled=False
    cfg.server.host='127.0.0.1';cfg.server.port=18849
    server=build_server_from_config(cfg,str(ROOT/'assets/dinov3-policy'),device='cuda:0')
    enc=server.engine.architecture.video_backbone.video_encoder
    state=load_file(ROOT/'assets/dinov3-study/encoder.safetensors')
    prefix='video_backbone.video_encoder.'
    subset={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    assert set(subset)==set(enc.state_dict())
    assert all(torch.equal(enc.state_dict()[k].cpu(),v.to(enc.state_dict()[k].dtype)) for k,v in subset.items())
    # Deterministic synthetic pixels check full-policy vs standalone numerical path.
    image=Image.fromarray(np.random.default_rng(1973).integers(0,256,(384,320,3),dtype=np.uint8))
    with torch.inference_mode():
        pixels=enc.preprocess_video([image]);z=enc.batch_encode(pixels)
    if not args.adapter:
        torch.save({'pixels':pixels.cpu(),'latent':z.cpu(),
                    'dtypes':{k:str(v.dtype) for k,v in enc.state_dict().items()}},OUT/'encoder-reference.pt')
        (OUT/'encoder-reference.json').write_text(json.dumps({'published_encoder_weights_exact':True,
             'shape':list(z.shape),'dtype':str(z.dtype),'full_policy_reference':True},indent=2))
    counts={'calls':0,'max_change':0.,'adapter':args.adapter}
    if args.adapter:
        ck=torch.load(args.adapter,map_location='cpu',weights_only=True)
        adapter=Adapter(ck['kind']).cuda().eval();adapter.load_state_dict(ck['state_dict'],strict=True)
        adapter.requires_grad_(False)
        original=enc.batch_encode
        def adapted(video):
            latent=original(video)
            output=adapter(latent)
            assert torch.equal(output[:,:,1:],latent[:,:,1:])
            counts['calls']+=1
            counts['max_change']=max(counts['max_change'],float((output-latent).abs().max()))
            (OUT/'adapter-hook.json').write_text(json.dumps(counts,indent=2))
            return output
        enc.batch_encode=adapted
    print('ROBUST_SERVER_READY',json.dumps(counts),flush=True)
    try:server.run(host='127.0.0.1',port=18849)
    finally:server.shutdown()

if __name__=='__main__':main()
