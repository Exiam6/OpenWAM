import argparse, hashlib, json, logging, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from robustness_common import Adapter
from norm_common import ARMS, intervene
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/norm-20260921'
sys.path.insert(0,str(ROOT/'OpenWAM'))
from openwam.deploy.server import build_server_from_config

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True);args=parser.parse_args()
    logging.basicConfig(level=logging.INFO);torch.manual_seed(42);torch.set_num_threads(4)
    cfg=OmegaConf.load(ROOT/'OpenWAM/configs/deploy.yaml');cfg.optimization.compile.enabled=False
    cfg.server.host='127.0.0.1';cfg.server.port=18848
    server=build_server_from_config(cfg,str(ROOT/'assets/dinov3-policy'),device='cuda:0')
    enc=server.engine.architecture.video_backbone.video_encoder
    weights=load_file(ROOT/'assets/dinov3-study/encoder.safetensors');prefix='video_backbone.video_encoder.'
    subset={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
    assert set(subset)==set(enc.state_dict())
    assert all(torch.equal(enc.state_dict()[k].cpu(),v.to(enc.state_dict()[k].dtype)) for k,v in subset.items())
    assert enc._out_norm.eps==1e-6 and not enc._out_norm.elementwise_affine
    ckpath=ROOT/'results/robustness-20260921/adapter-selected.pt'
    assert hashlib.sha256(ckpath.read_bytes()).hexdigest()=='f0d0cf8eeae4b87e3d16a742c8733e05b82ec737a445dec6cd18efedd3092586'
    ck=torch.load(ckpath,map_location='cpu',weights_only=True)
    adapter=Adapter(ck['kind']).cuda().eval().requires_grad_(False);adapter.load_state_dict(ck['state_dict'],strict=True)
    original=enc.batch_encode
    counts={'arm':args.arm,'calls':0,'max_change':0.,'future_slots_exact':True,'published_encoder_weights_exact':True,'scope':'cumulative for this arm server across reference/historical/primary as applicable'}
    def encode(video):
        z=original(video)
        output=intervene(z,args.arm,adapter,enc._apply_feature_norm)
        counts['calls']+=1;counts['max_change']=max(counts['max_change'],float((output-z).abs().max()))
        tmp=OUT/f'{args.arm}-hook.tmp';tmp.write_text(json.dumps(counts,indent=2));tmp.replace(OUT/f'{args.arm}-hook.json')
        return output
    enc.batch_encode=encode
    print('NORM_SERVER_READY',json.dumps(counts),flush=True)
    try:server.run(host='127.0.0.1',port=18848)
    finally:server.shutdown()
if __name__=='__main__':main()
