from pathlib import Path
import json,hashlib,datetime,importlib.util,time
D=Path(__file__).parent;V=D.parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(16<<20),b''):h.update(data)
 return h.hexdigest()

plan=json.loads((D/'plan.json').read_text())
assert plan['scope']=='encoder_precision_diagnostic_no_release'
assert sha(__file__)==plan['diagnostic_code_sha256']
for path,digest in plan['frozen_files'].items():assert sha(path)==digest,path
assert not (V/'full-training-release.json').exists()
import torch,numpy as np
from PIL import Image
from safetensors.torch import load_file
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38,WanVideoVAE
from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=D/'observations';out.mkdir(exist_ok=False);start=time.monotonic();result=dict(diagnostic_only=True,full_training_released=False,original_gate_status='failed',completed=False,rows=[])
try:
 spec=importlib.util.spec_from_file_location('isolated_precision_fixed_vae',plan['fixed_module']);fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)
 model=WanVideoVAE38().to(dtype=torch.bfloat16);values=load_file(str(V/'assets/wan-vae48.safetensors'),device='cpu');model.load_state_dict(values,strict=True);del values
 model.requires_grad_(False).eval().to('cuda');encoder=WanVideoVAEEncoder(model)
 manifest=json.loads((V/'attempt-data-restored/window-manifest.json').read_text());sources={x['episode']:x for x in manifest['sources']}
 cache=torch.load(V/'vae-cache/cache.pt',map_location='cpu',weights_only=False,mmap=True)
 earlier=torch.load(V/'deployment-diagnostic-0807/observations/adjust_bottle-forward-inputs.pt',map_location='cpu',weights_only=False)
 for task in ['adjust_bottle','handover_block','place_object_basket']:
  m=next(x for x in manifest['windows'] if x['task']==task and x['split']=='train');source=sources[m['episode']];assert sha(source['path'])==source['sha256']
  with np.load(source['path'],allow_pickle=False) as f:frame=Image.fromarray(f['images'][m['rgb_indices'][0]])
  pixels=encoder.preprocess_video([frame])
  with torch.no_grad(),torch.autocast('cuda',enabled=False):off=WanVideoVAE.batch_encode(model,pixels,'cuda').cpu()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   on=WanVideoVAE.batch_encode(model,pixels,'cuda').cpu()
   corrected=fixed.WanVideoVAE.batch_encode(model,pixels,'cuda').cpu()
   repeated=WanVideoVAE.batch_encode(model,pixels,'cuda').cpu()
  truth=cache['vae'][m['cache_index']:m['cache_index']+1,:,:1]
  arrays={'original_no_autocast':off,'original_outer_autocast':on,'fixed_outer_autocast':corrected,'original_repeated_autocast':repeated,'cache':truth}
  path=out/(task+'.pt');torch.save(arrays,path)
  row=dict(task=task,cache_index=m['cache_index'],arrays=str(path),arrays_sha256=sha(path),all_finite=all(bool(torch.isfinite(x).all()) for x in arrays.values()),original_off_equals_cache=torch.equal(off,truth),original_on_vs_cache=float((on.float()-truth.float()).abs().max()),fixed_on_equals_cache=torch.equal(corrected,truth),fixed_on_vs_cache=float((corrected.float()-truth.float()).abs().max()),original_on_repeats_exactly=torch.equal(on,repeated))
  if task=='adjust_bottle':row['original_on_equals_prior_image_forward']=torch.equal(on,earlier['image']['first_frame_latents'])
  result['rows'].append(row);(out/'progress.json').write_text(json.dumps(result,indent=2))
 result['completed']=True
except BaseException as exc:result['error']=repr(exc);raise
finally:
 result['seconds']=time.monotonic()-start;result['time']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
