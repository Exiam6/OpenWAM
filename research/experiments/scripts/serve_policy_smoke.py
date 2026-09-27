import logging,sys,torch
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'OpenWAM'))
from openwam.deploy.server import build_server_from_config
logging.basicConfig(level=logging.INFO);torch.manual_seed(42)
cfg=OmegaConf.load(ROOT/'OpenWAM/configs/deploy.yaml');cfg.optimization.compile.enabled=False
cfg.server.host='127.0.0.1';cfg.server.port=18848
server=build_server_from_config(cfg,str(ROOT/'assets/dinov3-policy'),device='cuda:0')
try:server.run(host='127.0.0.1',port=18848)
finally:server.shutdown()
