#!/usr/bin/env python3
"""Two real demonstration observations through the unmodified published policy.
No action is executed; this is an inference integration smoke test, not an eval.
"""
import hashlib,json,logging,os,subprocess,sys,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
from PIL import Image
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'OpenWAM'))
from openwam.deploy.server import build_server_from_config
from benchmarks.utils.action_conversion import robotwin_endpose_to_eef20d,eef20d_to_ee16d
from benchmarks.robotwin.prompt_template import format_prompt_for_inference
OUT=ROOT/'results/policy-smoke-20260921';OUT.mkdir(parents=True,exist_ok=True)

def guard():
 uid=os.environ['CUDA_VISIBLE_DEVICES']
 ecc=subprocess.check_output(['nvidia-smi','-i',uid,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip()
 assert ecc=='0'
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
 assert not [r for r in rows if r.split(',')[0].strip()==uid and int(r.split(',')[1])!=os.getpid()]

def main():
 logging.basicConfig(level=logging.INFO);guard();torch.manual_seed(42)
 assets=ROOT/'assets/dinov3-policy'
 assert (assets/'download-provenance.json').is_file(),'Full checkpoint download and SHA256 verification required'
 config=OmegaConf.load(ROOT/'OpenWAM/configs/deploy.yaml')
 config.optimization.compile.enabled=False
 config.server.host='127.0.0.1'
 start=time.perf_counter();server=build_server_from_config(config,str(assets),device='cuda:0')
 loaded=time.perf_counter()-start
 # Use a TRAIN episode according to the fixed split, no test labels or predictions.
 episode=int(np.random.default_rng(42).permutation(50)[0])
 files=list((ROOT/'data/pick_dual_bottles').rglob(f'episode{episode}.hdf5'));assert len(files)==1
 result={'scope':'published policy inference only; no simulated or physical actions executed',
         'task':'pick_dual_bottles','episode':episode,'episode_split':'train','seed':42,
         'checkpoint_revision':'af1595c8ee54955116abbc0b0ffa17fa48deb691',
         'checkpoint_sha256':'2badc3aaa2cf03054e90438e5798bd5beaf1744933284d66b53fe3611fabc841',
         'compile_enabled':False,'denoise_steps':10,'denoise_mode':'sync','dit_cache_enabled':True,
         'official_compile_protocol_matched':False,'load_seconds':loaded,'observations':[]}
 with h5py.File(files[0],'r') as f:
  for index in [0,len(f['endpose/left_endpose'])//2]:
   guard();server.reset();torch.manual_seed(42)
   images={dst:Image.fromarray(cv2.imdecode(np.frombuffer(bytes(f[f'observation/{src}/rgb'][index]),np.uint8),cv2.IMREAD_COLOR)) for src,dst in [('head_camera','head_camera'),('left_camera','left_wrist_camera'),('right_camera','right_wrist_camera')]}
   state=robotwin_endpose_to_eef20d(f['endpose/left_endpose'][index],f['endpose/right_endpose'][index],f['endpose/left_gripper'][index],f['endpose/right_gripper'][index])
   assert state.shape==(20,) and np.isfinite(state).all()
   obs={'images':images,'state':state.tolist(),'prompt':format_prompt_for_inference('Pick up both bottles.')}
   torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
   with torch.inference_mode():prediction=server.predict(obs)
   torch.cuda.synchronize();elapsed=time.perf_counter()-t
   action=np.asarray(prediction['action']);assert action.shape==(20,) and np.isfinite(action).all()
   converted=eef20d_to_ee16d(action);assert converted.shape==(16,) and np.isfinite(converted).all()
   row={'frame':index,'inference_seconds':elapsed,'peak_torch_allocated_gb':torch.cuda.max_memory_allocated()/1e9,
        'input_state':state.tolist(),'action20':action.tolist(),'converted_action16':converted.tolist(),
        'finite_action':True,'action_shape':[20],'grippers':action[[9,19]].tolist(),
        'converted_quaternion_norms':[float(np.linalg.norm(converted[3:7])),float(np.linalg.norm(converted[11:15]))]}
   result['observations'].append(row);(OUT/'result.json').write_text(json.dumps(result,indent=2));print('OBSERVATION',json.dumps(row),flush=True)
 server.shutdown();result['complete']=True;result['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print('POLICY SMOKE COMPLETE',flush=True)

if __name__=='__main__':main()
