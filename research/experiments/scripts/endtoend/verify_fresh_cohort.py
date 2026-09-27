"""Verify all150 experts, raw hashes, first-scene identity and exclusion before learned-policy evaluation."""
import json,hashlib
from pathlib import Path
import h5py,numpy as np
from pipeline_common import *
guard();assert not (S/'fresh-cohort-integrity.json').exists()
# Prior reference includes both storage groups; explicitly add the latest60 cohort.
old=json.loads((R/'results/confirmation-20260923/identity-reference.json').read_text());known=set(old['hashes'])
for item in json.loads((R/'results/confirmation-20260923/sample-manifest.json').read_text()):known.update([item['episode_sha256'],item['head_encoded_sha256']])
seen_files=set();seen_initial=set();seen_poses=set();rows=[];manifests={}
for task in ['adjust_bottle','handover_block','place_object_basket']:
 p=E/'scenes'/task/'manifest.json';m=json.loads(p.read_text());assert m['complete'] and len(m['accepted'])==50;manifests[str(p)]=sha(p)
 expected=[r for r in m['attempts'] if r.get('accepted')];assert expected==m['accepted']
 for r in m['accepted']:
  assert r['exit_code']==0 and r['expert_plan_success'] and r['expert_replay_success']
  path=Path(r['hdf5']);h=sha(path);assert h==r['hdf5_sha256'] and h not in known and h not in seen_files;seen_files.add(h)
  with h5py.File(path,'r') as f:
   encoded=bytes(f['observation/head_camera/rgb'][0]);img=hashlib.sha256(encoded).hexdigest();assert img not in known and img not in seen_initial;seen_initial.add(img)
   assert len(f['endpose/left_endpose'])==r['frames']
  base=path.parent.parent;initial=json.loads((base/'initial-replay.json').read_text());plan=json.loads((base/'initial-plan.json').read_text());assert initial['eval_mode'] and plan['eval_mode']
  for key in initial['poses']:
   for c in ['p','q']:np.testing.assert_allclose(initial['poses'][key][c],plan['poses'][key][c],rtol=0,atol=1e-6)
  # Exact pose signature gives an additional within-cohort duplicate check, not semantic novelty proof.
  posehash=hashlib.sha256(json.dumps({'task':task,'poses':initial['poses']},sort_keys=True).encode()).hexdigest();assert posehash not in seen_poses;seen_poses.add(posehash)
  rows.append({'task':task,'seed':r['seed'],'episode_sha256':h,'head_initial_encoded_sha256':img,'initial_poses_sha256':posehash})
write(S/'fresh-cohort-integrity.json',{'passed':True,'time':now().isoformat(),'episodes':150,'prior_hashes':len(known),'manifests_sha256':manifests,'rows':rows,'scope':'new seed ranges excluded onbothgroups; exact episode/encodedinitialframe and pose duplicate checks; no claim of unseen object categories or tasks'})
