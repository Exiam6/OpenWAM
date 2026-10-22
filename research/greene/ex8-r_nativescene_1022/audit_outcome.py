"""Read-only conservative terminal audit; never changes dispatch/claims."""
import json,sys,hashlib,math
from pathlib import Path
from integration import outcome
from native_adapter import pci
R=Path('/home/zifanz4/openwam-runtime');F=R/'ex8-r_nativescene_1022';J='ex8r-native-scene'
def audit():
 def read(p):return json.loads(p.read_text()) if p.exists() else None
 s,w,c=read(F/'result.json'),read(R/'nightly-gpu/runs'/J/'result.json'),read(R/'nightly-gpu/claims'/(J+'.json'))
 result=outcome(s,w,c)
 if result['GPU_diagnostic_validated']:
  try:
   d=s['native_details'];dev=d['device'];text=d['prewarm_stdout']
   proof=d['device_access']
   assert proof['landlock_abi']>=3 and proof['thread_only'] is True
   assert pci(proof['expected_pci'])==pci(d['expected_pci']) and len(proof['allowed_device_nodes'])==3
   assert 'prewarmed CUDA/Curobo before SAPIEN import' in text and 'CUDA/Curobo prewarm failed:' not in text
   assert dev['cuda_id']==0 and pci(dev['pci'])==pci(d['expected_pci'])
   assert type(d['instance_boundary_create']) is int and d['instance_boundary_create']==0
   scene=d['native_scene'];assert scene['policy_loaded'] is False and scene['expert_rollout'] is False and scene['scored_episodes']==0
   assert scene['task']=='adjust_bottle' and scene['mode']=='demo_clean' and scene['candidate_seed']==100000
   assert pci(scene['device']['pci'])==pci(d['expected_pci'])
   assert all(math.isfinite(x) for x in scene['observation']['state'])
   assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in scene['image_sha256'].items())
   assert set(scene['image_sha256'])=={v['path'] for v in scene['observation']['images'].values()}
   assert len(scene['observation']['state'])==20 and set(scene['observation']['images'])=={'head_camera','left_camera','right_camera'}
   assert all(Path(v['path']).is_file() for v in scene['observation']['images'].values())
   assert d['headless_init'] in [0,1] and d['headless_support']==0
   assert all(d[n]==1 for n in ['headless_loader','headless_getproc','headless_dispatcher','headless_layer_symbol','headless_layer_count','headless_layer_data'])
  except Exception as exc:result.update(GPU_diagnostic_validated=False,assessment='native evidence failed: '+repr(exc))
 return result
if __name__=='__main__':print(json.dumps(audit(),indent=2))
