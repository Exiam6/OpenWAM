"""Original frozen policy, fresh server per round; no representation hook."""
import argparse,json,logging,sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'OpenWAM'))
from openwam.deploy.server import build_server_from_config

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,choices=[0,1],required=True);a=parser.parse_args()
    out=ROOT/'results/repeatability-20260921/closedloop'
    logging.basicConfig(level=logging.INFO);torch.manual_seed(42);torch.set_num_threads(4)
    cfg=OmegaConf.load(ROOT/'OpenWAM/configs/deploy.yaml');cfg.optimization.compile.enabled=False;cfg.server.host='127.0.0.1';cfg.server.port=18848
    server=build_server_from_config(cfg,str(ROOT/'assets/dinov3-policy'),device='cuda:0');server._init_policy()
    enc=server.engine.architecture.video_backbone.video_encoder
    weights=load_file(ROOT/'assets/dinov3-study/encoder.safetensors');prefix='video_backbone.video_encoder.'
    subset={k[len(prefix):]:v for k,v in weights.items()if k.startswith(prefix)}
    assert set(subset)==set(enc.state_dict())
    assert all(torch.equal(enc.state_dict()[k].cpu(),v.to(enc.state_dict()[k].dtype))for k,v in subset.items())
    assert not server._policy._async and server._policy._executor.inference_horizon is None
    OmegaConf.save(server.cfg,out/f'round{a.round}-effective-config.yaml')
    (out/f'round{a.round}-server-check.json').write_text(json.dumps({'encoder_weights_exact':True,'sync_executor':True,'inference_horizon':None,'representation_hook':False},indent=2)+'\n')
    print('TRAJECTORY_SERVER_READY',flush=True)
    try:server.run(host='127.0.0.1',port=18848)
    finally:server.shutdown()
if __name__=='__main__':main()
