import ast,base64,datetime,hashlib,json,random,sys
from pathlib import Path
import numpy as np,torch,h5py
from omegaconf import OmegaConf
import openwam.deploy.engine as engine
from openwam.deploy.server import build_server_from_config
from benchmarks.utils.action_conversion import robotwin_endpose_to_eef20d
S=Path(__file__).parent;plan=json.loads((S/'protocol.json').read_text());O=Path(plan['output']);src=Path(engine.__file__);text=src.read_text()
assert hashlib.sha256(text.encode()).hexdigest()==plan['engine_sha256']
node=next(n for n in ast.parse(text.replace('            evicted_key, _ = self.popitem(last=False)','            evicted_key = next(iter(self))\n            super().__delitem__(evicted_key)')).body if isinstance(n,ast.ClassDef) and n.name=='_BoundedPromptEmbedCache')
ns=dict(engine.__dict__);exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated patched cache>','exec'),ns);Fixed=ns['_BoundedPromptEmbedCache'];Original=engine._BoundedPromptEmbedCache
cfg=OmegaConf.load(Path(plan['source'])/'configs/deploy.yaml');cfg.optimization.compile.enabled=False;cfg.optimization.dit_cache.enabled=False;cfg.inference.denoise_steps=10;cfg.inference.inference_horizon=8
with h5py.File(plan['training_hdf5'],'r') as f:
 images={target:base64.b64encode(bytes(f['observation/'+source+'/rgb'][0])).decode() for source,target in [('head_camera','head_camera'),('left_camera','left_wrist_camera'),('right_camera','right_wrist_camera')]}
 state=robotwin_endpose_to_eef20d(f['endpose/left_endpose'][0],f['endpose/right_endpose'][0],f['endpose/left_gripper'][0],f['endpose/right_gripper'][0]).tolist()
payload={'images':images,'state':state,'prompt':plan['prompt']}
torch.set_num_threads(4);server=build_server_from_config(cfg,plan['checkpoint_dir'],device='cuda:0');outputs={};cache_records={}
def predict(label):
 assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(plan['deadline'])
 assert not Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
 server.reset();server.engine._vace_cache.clear();random.seed(1771);np.random.seed(1771);torch.manual_seed(1771);torch.cuda.manual_seed_all(1771)
 result=server.predict(payload);x=np.asarray(result['action']);assert np.isfinite(x).all() and x.shape[-1]==20
 outputs[label]=x.tolist();cache_records[label]=list(server.engine._prompt_embed_cache)
 assert plan['prompt'] in server.engine._prompt_embed_cache
 print('PREDICT',label,x.shape,flush=True)
 (O/'progress.json').write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'completed':list(outputs)},indent=2)+'\n')
try:
 server.engine._prompt_embed_cache=Original(32);predict('original_cold');predict('original_hit')
 engine._BoundedPromptEmbedCache=Fixed;server.engine._prompt_embed_cache=Fixed(32);predict('fixed_cold');predict('fixed_hit')
 cache=server.engine._prompt_embed_cache;value=cache[plan['prompt']]
 # Insert distinct sentinel aliases solely to force eviction; they are never inference prompts.
 for i in range(32):cache['__cache_capacity_sentinel_'+str(i)]=value
 assert len(cache)==32 and plan['prompt'] not in cache
 predict('fixed_after_eviction')
 ref=np.asarray(outputs['original_cold']);differences={k:float(np.max(np.abs(np.asarray(v)-ref))) for k,v in outputs.items()};passed=all(v==0 for v in differences.values())
 (O/'result.json').write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'passed':passed,'action_max_abs_differences':differences,'actions':outputs,'inference_calls':5,'eviction_capacity':32,'training_frame_only':True,'heldout_episodes':0,'limits':['one S-VAE295M checkpoint and one training frame','direct native prediction, not WebSocket or simulator','sentinel alias insertion forces eviction; not33 real language prompts','not a success-rate evaluation'],'peak_cuda_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n');assert passed,differences
finally:server.shutdown()
