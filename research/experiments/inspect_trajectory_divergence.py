"""Describe all first divergences; make no change to frozen comparison metrics."""
import base64,gzip,io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/data02/zifanz4/openwam-experiments');RUN=ROOT/'results/repeatability-20260921/closedloop'
sys.path.insert(0,str(ROOT/'scripts'))
from repeat_common import difference
s=json.loads((RUN/'summary.json').read_text());rows=[]
for pair in s['paired_trajectories']:
 cond=pair['condition'];seed=pair['seed'];traces=[[json.loads(x)for x in(RUN/f'r{rid}-{cond}'/f'seed-{seed}-trace.jsonl').read_text().splitlines()]for rid in[0,1]]
 first=pair['first_different_step'];earliest=min([v for k,v in first.items()if v is not None],default=None)
 row={'condition':cond,'seed':seed,'first_recorded_difference':earliest,'first_difference_fields':[k for k,v in first.items()if v is not None and v==earliest],'first_step_numeric_differences':{},'action_generation_context':None}
 if earliest is not None:
  a,b=[t[earliest]for t in traces]
  row['first_step_numeric_differences']={'eef':difference(np.asarray(a['eef_state'],np.float32),np.asarray(b['eef_state'],np.float32)),'server_action':difference(np.asarray(a['server_action'],np.float32),np.asarray(b['server_action'],np.float32)),'joint_max_abs':float(np.max(np.abs(np.array(a['joint_state'])-np.array(b['joint_state']))))}
  row['raw_camera_equal_at_first_difference']={k:a['raw_cameras'][k]==b['raw_cameras'][k]for k in a['raw_cameras']}
  payloads=[json.loads(gzip.decompress((RUN/f'r{rid}-{cond}'/t[earliest]['request_file']).read_bytes()))for rid,t in enumerate(traces)]
  row['policy_camera_pixel_differences_at_first_difference']={}
  for camera in payloads[0]['images']:
   images=[np.asarray(Image.open(io.BytesIO(base64.b64decode(p['images'][camera]))).convert('RGB'))for p in payloads]
   x,y=images;delta=np.abs(x.astype(np.int16)-y.astype(np.int16))
   row['policy_camera_pixel_differences_at_first_difference'][camera]={'changed_rgb_channels':int(np.count_nonzero(x!=y)),'total_rgb_channels':int(x.size),'max_byte_difference':int(delta.max()),'mean_absolute_byte_difference':float(delta.mean()),'changed_pixel_fraction':float(np.any(x!=y,axis=-1).mean())}
 step=first['server_action']
 if step is not None:
  # Inferred from frozen synchronous full32-action buffer, not a new server trace.
  boundary=(step//32)*32;a,b=[t[boundary]for t in traces];x,y=[t[step]for t in traces]
  row['action_generation_context']={'first_different_action_index':step,'inferred_generation_boundary':boundary,'inferred_buffer_offset':step%32,'boundary_request_equal':a['request_sha256']==b['request_sha256'],'current_request_equal':x['request_sha256']==y['request_sha256'],'difference_at_first_changed_action':difference(np.asarray(x['server_action'],np.float32),np.asarray(y['server_action'],np.float32)),'boundary_is_inferred_not_directly_traced':True}
 rows.append(row)
result={'role':'post-run descriptive first-divergence inspection; all six pairs retained','pairs':rows,'caveat':'Temporal ordering narrows the investigation; it does not identify a specific physics, renderer, IK or model cause.'}
(RUN/'first-divergence-inspection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
