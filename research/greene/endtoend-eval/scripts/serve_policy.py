"""Native server plus explicit per-episode inference seeds for paired evaluation.

Greene adaptation of research/experiments/scripts/endtoend/serve_policy.py:
deploy.yaml and code come from the endtoend OpenWAM snapshot; the checkpoint is
loaded from its Greene view directory (see build_view.py). Serving overrides are
unchanged: compile off, DiT cache off, denoise_steps 10, inference_horizon 8.
"""
import argparse,json,random,sys
from pathlib import Path
import numpy as np,torch
from omegaconf import OmegaConf
import openwam
from openwam.deploy.server import build_server_from_config
E=Path('/scratch/zz4330/openwam-runtime/endtoend-root/endtoend-20260923')
assert Path(openwam.__file__).resolve().is_relative_to((E/'OpenWAM').resolve()),openwam.__file__
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--log',required=True);a=p.parse_args()
cfg=OmegaConf.load(E/'OpenWAM/configs/deploy.yaml');cfg.optimization.compile.enabled=False;cfg.optimization.dit_cache.enabled=False;cfg.inference.denoise_steps=10;cfg.inference.inference_horizon=8;cfg.server.host='127.0.0.1';cfg.server.port=a.port
torch.set_num_threads(4);server=build_server_from_config(cfg,str(Path(a.checkpoint).parent),device='cuda:0',ckpt_name=Path(a.checkpoint).name);original=server.predict

def predict(obs):
 step=obs.pop('_study_step');seed=obs.pop('_study_action_seed')
 assert isinstance(step,int) and step>=0 and isinstance(seed,int) and 0<=seed<2**32
 if step==0:
  server.reset();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  with Path(a.log).open('a') as f:f.write(json.dumps({'episode_seed':seed,'reset_before_prediction':True})+'\n')
 result=original(obs)
 cache=server.engine._prompt_embed_cache
 assert cache is not None and len(cache)<=32
 if step==0:
  with Path(a.log).open('a') as f:f.write(json.dumps({'cache_entries_after_prediction':len(cache),'cache_capacity':32})+'\n')
 return result
server.predict=predict
print('ENDTOEND_SERVER_READY',flush=True)
try:server.run(host='127.0.0.1',port=a.port)
finally:server.shutdown()
