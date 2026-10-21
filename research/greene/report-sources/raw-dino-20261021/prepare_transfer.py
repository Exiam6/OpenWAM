import sys,json,pathlib,concurrent.futures
ROOT=pathlib.Path('/home/zifanz4');R=ROOT/'openwam-runtime';out=R/'ex6-c_resourcerecovery_1020/paired-report';out.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'openwam-vae-control/research/greene/ex5-a_vaebaseline_1018'))
from paired_video import encode
state=json.loads((R/'ex6-c_resourcerecovery_1020/transfer-status.json').read_text());specs=[]
for task,t in state['tasks'].items():
 arms=t['arms'];left=arms['raw768_native'];right=arms['vae48_native']
 if not all(a['verified']==a['planned'] for a in arms.values()):continue
 for l in left['rows']:
  rr=next(x for x in right['rows'] if x['scene_seed']==l['scene_seed']);seed=l['scene_seed'];key=f'{task}-{seed}'
  sides=[dict(result_path=x['source'],result_sha256=x['sha256'],trace_sha256=x['trace_sha256'],result=json.loads(pathlib.Path(x['source']).read_text())) for x in [l,rr]]
  specs.append(dict(key=key,kind='model',comparison='transfer_raw_minus_vae',seed=42,task=task,scene_seed=seed,labels=['Raw','VAE'],routes=['raw768_native','vae48_native'],left=sides[0],right=sides[1],output=str(out/f'{key}.json')))
def one(s):
 p=pathlib.Path(s['output']);v=json.loads(p.read_text()) if p.exists() else encode(s);print(v['key'],flush=True);return v
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(one,specs))
(out/'index.json').write_text(json.dumps(result,indent=2));print('COMPLETE',len(result),flush=True)
