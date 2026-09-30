"""Check the Greene copy of the sealed endtoend cohort against the frozen fresh-cohort-integrity.json.

Same identity checks as research/experiments/scripts/endtoend/verify_fresh_cohort.py (episode hdf5,
encoded initial head frame, initial pose signature) but compared to the frozen values rather than
recomputed from scratch, with /data02 paths remapped to the Greene view. Writes <output> with passed:true.
"""
import argparse,datetime,hashlib,json
from pathlib import Path
import h5py
REPO=Path('/scratch/zz4330/OpenWAM');E=Path('/scratch/zz4330/openwam-runtime/endtoend-root/endtoend-20260923')
FROZEN=REPO/'research/experiments/studies/endtoend-20260923/fresh-cohort-integrity.json'
OLD='/data02/zifanz4/openwam-experiments/endtoend-20260923/'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<24),b''):h.update(b)
 return h.hexdigest()
def local(p):assert p.startswith(OLD),p;return E/p[len(OLD):]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 frozen=json.loads(FROZEN.read_text());assert frozen['passed'] and frozen['episodes']==150==len(frozen['rows'])
 for p,want in frozen['manifests_sha256'].items():assert sha(local(p))==want,p
 rows={(r['task'],int(r['seed'])):r for r in frozen['rows']};checked=0;scene_files={}
 for task in ['adjust_bottle','handover_block','place_object_basket']:
  m=json.loads((E/'scenes'/task/'manifest.json').read_text());assert m['complete'] and len(m['accepted'])==50
  for r in m['accepted']:
   want=rows[(task,int(r['seed']))];path=local(r['hdf5'])
   assert sha(path)==r['hdf5_sha256']==want['episode_sha256'],path
   with h5py.File(path,'r') as f:assert hashlib.sha256(bytes(f['observation/head_camera/rgb'][0])).hexdigest()==want['head_initial_encoded_sha256'],path
   base=path.parent.parent;initial=json.loads((base/'initial-replay.json').read_text())
   assert hashlib.sha256(json.dumps({'task':task,'poses':initial['poses']},sort_keys=True).encode()).hexdigest()==want['initial_poses_sha256'],base
   # Files evaluate_policy reads per scene; not in the frozen record, hashed here for provenance.
   for n in ['collection-config.json','initial-replay.json','initial-head_camera.png','initial-left_camera.png','initial-right_camera.png']:scene_files[str(base/n)]=sha(base/n)
   checked+=1
 assert checked==150
 out=Path(a.output);tmp=out.with_suffix('.tmp')
 tmp.write_text(json.dumps({'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),'source':str(FROZEN),'source_sha256':sha(FROZEN),'episodes_checked':checked,'scene_files_sha256':scene_files},indent=2)+'\n');tmp.replace(out)
 print('COHORT OK',checked,flush=True)
if __name__=='__main__':main()
