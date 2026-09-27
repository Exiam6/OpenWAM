"""Strict native checkpoint reload, observation parity and deterministic action smoke test."""
import argparse,gc,hashlib,json,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
from PIL import Image
from omegaconf import OmegaConf
from openwam.deploy.server import build_server_from_config
from openwam.dataloader.registry import build_dataset
R=Path('/data02/zifanz4/openwam-experiments');E=R/'endtoend-20260923'
p=argparse.ArgumentParser();p.add_argument('--route',required=True);p.add_argument('--profile',default='profiles-v3');a=p.parse_args()
out=E/'deploy-preflight'/a.route;out.mkdir(parents=True,exist_ok=True);assert not(out/'result.json').exists()
dirs=sorted((E/a.profile/a.route/'output').glob('*'));assert len(dirs)==1;ckpt=dirs[0]
assert json.loads((E/a.profile/a.route/'exit.json').read_text())['returncode']==0
cfg=OmegaConf.load(E/'OpenWAM/configs/deploy.yaml');cfg.optimization.compile.enabled=False;cfg.optimization.dit_cache.enabled=False;cfg.inference.inference_horizon=8;cfg.inference.denoise_steps=10
begin=time.monotonic();torch.set_num_threads(4)
server=build_server_from_config(cfg,str(ckpt),device='cuda:0');server._init_policy()
training=OmegaConf.load(ckpt/'config.yaml');dataset=build_dataset(training.dataloader);ds=dataset._sub_datasets[0];sample=ds._build_sample(0,0)
images={}
with h5py.File(ds._episode_files[0],'r') as f:
 state=ds._read_raw_actions(f,0,1)[0]
 for src,dst in [('head_camera','head_camera'),('left_camera','left_wrist_camera'),('right_camera','right_wrist_camera')]:images[dst]=Image.fromarray(cv2.imdecode(np.frombuffer(bytes(f[f'observation/{src}/rgb'][0]),dtype=np.uint8),cv2.IMREAD_COLOR))
obs={'images':images,'prompt':sample['prompt'],'state':state.tolist()}
processed=server._obs_preprocessor.preprocess(dict(obs));assert np.array_equal(np.asarray(processed['image']),np.asarray(sample['video'][0]))
answers=[]
for _ in range(2):
 server.reset();torch.manual_seed(1771);torch.cuda.manual_seed_all(1771)
 answer=server.predict(dict(obs));act=np.asarray(answer['action']);assert act.size==20 and np.isfinite(act).all(),answer;answers.append(answer)
assert np.array_equal(np.asarray(answers[0]['action']),np.asarray(answers[1]['action']))
result={'passed':True,'route':a.route,'checkpoint':str(ckpt),'strict_native_deploy_load':True,'native_multiview':bool(training.dataloader.multiview),'train_deploy_composite_pixels_equal':True,'repeat_reset_action_equal':True,'action_dim':20,'denoise_steps':10,'replan_horizon':8,'seconds':time.monotonic()-begin,'first_latency_ms':answers[0]['latency_ms'],'warm_latency_ms':answers[1]['latency_ms'],'training_episode_used':str(ds._episode_files[0]),'heldout_scenes_used':False}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);server.shutdown()
