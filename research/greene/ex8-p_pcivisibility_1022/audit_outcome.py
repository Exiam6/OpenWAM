"""Read-only conservative terminal audit; never changes dispatch/claims."""
import json,sys
from pathlib import Path
from integration import outcome
from native_adapter import pci
R=Path('/home/zifanz4/openwam-runtime');F=R/'ex8-p_pcivisibility_1022';J='ex8p-native-pci-visibility'
def audit():
 def read(p):return json.loads(p.read_text()) if p.exists() else None
 s,w,c=read(F/'result.json'),read(R/'nightly-gpu/runs'/J/'result.json'),read(R/'nightly-gpu/claims'/(J+'.json'))
 result=outcome(s,w,c)
 if result['GPU_diagnostic_validated']:
  try:
   d=s['native_details'];dev=d['device'];text=d['prewarm_stdout']
   from pci_visibility import selector
   assert d['DRI_PRIME']==selector(d['expected_pci'])
   assert 'prewarmed CUDA/Curobo before SAPIEN import' in text and 'CUDA/Curobo prewarm failed:' not in text
   assert dev['cuda_id']==0 and pci(dev['pci'])==pci(d['expected_pci'])
   assert type(d['instance_boundary_create']) is int and d['instance_boundary_create']==0
   assert d['headless_init'] in [0,1] and d['headless_support']==0
   assert all(d[n]==1 for n in ['headless_loader','headless_getproc','headless_dispatcher','headless_layer_symbol','headless_layer_count','headless_layer_data'])
  except Exception as exc:result.update(GPU_diagnostic_validated=False,assessment='native evidence failed: '+repr(exc))
 return result
if __name__=='__main__':print(json.dumps(audit(),indent=2))
