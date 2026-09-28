import datetime,json
from pathlib import Path
BASE=Path('/home/zifanz4/openwam-runtime/temporal-new-20260928');S=BASE/'resume-v1'
def inventory():
 seen={};duplicates=[]
 roots=[BASE/'evaluation']+sorted((S/'runs').glob('*/evaluation'))
 for root in roots:
  for p in root.glob('*/*/*/seed-*'):
   if not p.is_dir():continue
   armseed,task,condition,scene=p.relative_to(root).parts;arm,seed=armseed.split('-seed');key=(arm,int(seed),task,condition,int(scene[5:]))
   f=p/'result.json';row=json.loads(f.read_text()) if f.exists() else {'status':'interrupted_without_result'}
   if key in seen:duplicates.append([list(key),seen[key]['path'],str(p)])
   seen[key]={'path':str(p),'status':row['status']}
 protocol=json.loads((BASE/'protocol.json').read_text());pending={}
 for seed in [42,43,44]:
  for arm in ['mean','learned']:
   for task in protocol['evaluation']['tasks']:
    manifest=json.loads((Path('/home/zifanz4/openwam-runtime/assets-source/temporal-20260926/scenes')/task/'manifest.json').read_text())
    for condition in protocol['evaluation']['conditions']:
     pending[(arm,seed,task,condition)]=[int(x['seed']) for x in manifest['accepted'] if (arm,seed,task,condition,int(x['seed'])) not in seen]
 return {'observed':datetime.datetime.now().astimezone().isoformat(),'completed':sum(x['status']=='complete' for x in seen.values()),'failures_or_interrupted':sum(x['status']!='complete' for x in seen.values()),'attempted':len(seen),'pending':sum(map(len,pending.values())),'duplicates':duplicates},pending
if __name__=='__main__':print(json.dumps(inventory()[0],indent=2))
