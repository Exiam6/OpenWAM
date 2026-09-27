"""Frozen matched codec confirmation, paired original noise and readout rules."""
import datetime,json,os,sys,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
from PIL import Image
R=Path(os.environ['WAM_ROOT']);B=R/'results/goalaux-20260922';L=R/'results/layers-20260922';O=B/'confirmation';F=B/'fresh-v2'
os.environ['LAYER_RUN']=str(L);sys.path.insert(0,str(R/'scripts/layers'));import common as c
from readout_v2 import visual,pool,predict
NEW=[f'{arm}_svae{s}' for arm in ['reconstruction_only','goal_auxiliary'] for s in [42,43,44]];ANCHORS=['L12_raw','L12_pca','L12_svae42','L12_svae43','L12_svae44'];CONDS=['clean','noise0.10','noise0.04']
def before(stage):
 freeze=json.loads((O/'freeze.json').read_text())
 for f,h in freeze['sha256'].items():assert c.sha(R/f)==h,f
 assert json.loads((O/'identity.json').read_text())['passed'];assert json.loads((O/'parity.json').read_text())['passed'];assert not (O/(stage+'-started.json')).exists()
 assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(freeze['deadline']);c.write(O/(stage+'-started.json'),{'time':datetime.datetime.now().astimezone().isoformat(),'freeze_sha256':c.sha(O/'freeze.json')})
 return freeze

def extract():
 freeze=before('extract');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;enc=c.encoder();features=[];rows=[];states=[];targets=[];refs=[];conds=[];start=time.monotonic();manifest=json.loads((F/'manifest.json').read_text());episodes=0
 for task in c.TASKS:
  for record in manifest['accepted'][task]:
   assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(freeze['deadline']);path=Path(record['hdf5']);assert c.sha(path)==record['hdf5_sha256'];seed=record['seed']
   with h5py.File(path,'r') as f:
    n=len(f['endpose/left_endpose'])
    for st in np.unique(np.linspace(0,n-33,12,dtype=int)):
     st=int(st);i=len(rows);lp=f['endpose/left_endpose'][st];rp=f['endpose/right_endpose'][st]
     states.append(np.concatenate([lp,rp,[f['endpose/left_gripper'][st],f['endpose/right_gripper'][st]]]))
     targets.append(np.concatenate([f['endpose/left_endpose'][st+4,:3]-lp[:3],f['endpose/right_endpose'][st+4,:3]-rp[:3],[f['endpose/left_gripper'][st+4],f['endpose/right_gripper'][st+4]]]))
     raw=bytes(f['observation/head_camera/rgb'][st]);arr=np.array(c.resize_head(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)));variants=[('clean',Image.fromarray(arr))]
     for sigma in [.1,.04]:
      for draw in range(3):
       rng=np.random.default_rng(20260922+seed*100000+st*10+draw);variants.append((f'noise{sigma:.2f}',Image.fromarray(np.clip(arr.astype(np.float32)+rng.normal(0,255*sigma,arr.shape),0,255).astype(np.uint8))))
     for name,im in variants:
      z=c.encode(enc,enc.preprocess_video([im]*9));features.append(z['L12'][0,:,:1].cpu().half());refs.append(i);conds.append(name)
     rows.append({'task':task,'seed':seed,'frame':st,'target_frame':st+4,'episode_sha256':record['hdf5_sha256'],'head_encoded_sha256':__import__('hashlib').sha256(raw).hexdigest()})
   episodes+=1;c.write(O/'progress.json',{'stage':'extracting','episodes':episodes,'frames':len(rows),'elapsed_seconds':time.monotonic()-start})
 del enc;torch.cuda.empty_cache();features=torch.stack(features);arrays=visual(features,'L12')
 for name in NEW:
  arm,seed=name.rsplit('_svae',1);ck=torch.load(B/'models'/arm/f'seed{seed}/final.pt',map_location='cpu',weights_only=False);model=c.SVAE(**ck['model_config']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['state_dict']);parts=[]
  with torch.inference_mode():
   for batch in features.split(8):
    with torch.autocast('cuda',dtype=torch.bfloat16):z=model.encode_mean(batch.cuda().float())
    parts.append(pool(z))
  arrays[name]=np.concatenate(parts);assert np.isfinite(arrays[name]).all();del model,parts;torch.cuda.empty_cache()
 assert len(rows)==720 and len(refs)==5040 and len(arrays)==11
 np.savez(O/'vectors.npz',**arrays,state=np.array(states,dtype=np.float32),target=np.array(targets,dtype=np.float32),reference=np.array(refs),condition=np.array(conds));c.write(O/'sample-manifest.json',rows)
 c.write(O/'extraction-complete.json',{'time':datetime.datetime.now().astimezone().isoformat(),'episodes':60,'frames':720,'image_variants':5040,'elapsed_seconds':time.monotonic()-start,'sha256':{n:c.sha(O/n) for n in ['vectors.npz','sample-manifest.json']},'fresh_scores_observed':False})
 print('EXTRACTION_COMPLETE',flush=True)

def metric(p,y,scale,episode,endpoint):
 err=((p.astype(float)-y.astype(float))/np.asarray(scale,dtype=float))**2;ids=np.unique(episode);assert len(ids)==20
 per=np.array([[err[episode==e,:6].mean(),err[episode==e,6:].mean()] if endpoint=='state' else err[episode==e].mean(0).tolist() for e in ids])
 return {'episode_ids':ids.tolist(),'per_episode_groups':per.tolist(),'E':float(per.mean()),'group_E':per.mean(0).tolist()}

def score():
 before('score');torch.set_num_threads(4);complete=json.loads((O/'extraction-complete.json').read_text())
 for f,h in complete['sha256'].items():assert c.sha(O/f)==h,f
 data=np.load(O/'vectors.npz');rows=json.loads((O/'sample-manifest.json').read_text());labels=json.loads((O/'labels.json').read_text());assert not labels['missing'];lookup={(r['task'],r['seed']):r['target_xy'] for r in labels['records']}
 ref=data['reference'];cond=data['condition'];tasks=np.array([r['task'] for r in rows]);ep=np.array([r['seed'] for r in rows]);frame=np.array([r['frame'] for r in rows]);state=data['state'];target=data['target'];assert ref.shape==(5040,)
 coefficients={'new':np.load(B/'readouts/coefficients.npz'),'state':np.load(L/'readout/coefficients.npz'),'goal':np.load(L/'initial-goal/coefficients.npz')}
 saved={'new':torch.load(B/'readouts/mlp.pt',map_location='cpu',weights_only=False),'state':torch.load(L/'readout/mlp/models.pt',map_location='cpu',weights_only=False),'goal':torch.load(L/'initial-goal/models.pt',map_location='cpu',weights_only=False)};constants=json.loads((L/'initial-goal/constants.json').read_text());results={};predictions={};scales_record={}
 for endpoint in ['state','goal']:
  for method in ['ridge','mlp']:
   for task in c.TASKS:
    for name in NEW+ANCHORS+(['proprio'] if endpoint=='state' else ['constant']):
     kind='new' if name in NEW else endpoint;key=f'{endpoint}/{task}/{name}' if kind=='new' else f'{task}/{name}'
     if name=='constant':scale=np.array(constants[task]['std']);mu=np.array(constants[task]['mean'])
     elif method=='ridge':
      models=[]
      for group in ([0,1] if endpoint=='state' else [0]):
       prefix=key+(f'/{group}' if kind=='new' or endpoint=='state' else '');models.append({k:coefficients[kind][f'{prefix}/{k}'] for k in ['xm','xs','ym','ys','w']})
      scale=np.concatenate([m['ys'] for m in models])
     else:
      xm,xs,ym,scale=[t.numpy() for t in saved[kind]['scales'][key]];model=torch.nn.Sequential(torch.nn.Linear(len(xm),128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,8 if endpoint=='state' else 2));model.load_state_dict(saved[kind]['states'][key]);model.eval()
     if endpoint=='goal':metric_scale=np.array(constants[task]['std'])
     else:metric_scale=scale
     scales_record[f'{endpoint}/{method}/{task}/{name}']=metric_scale.tolist()
     for variant in CONDS:
      mask=(tasks[ref]==task)&(cond==variant)
      if endpoint=='goal':mask&=frame[ref]==0
      ii=np.flatnonzero(mask);rr=ref[ii];y=target[rr] if endpoint=='state' else np.array([lookup[task,int(ep[j])] for j in rr],dtype=float)
      if name=='constant':pred=np.broadcast_to(mu,y.shape).copy()
      else:
       x=state[rr] if name=='proprio' else (np.concatenate([data[name][ii],state[rr]],1) if endpoint=='state' else data[name][ii])
       if method=='ridge':pred=np.concatenate([predict(m,x) for m in models],1)
       else:
        with torch.inference_mode():pred=model(torch.from_numpy((x-xm)/xs)).numpy()*scale+ym
      k=f'{endpoint}/{method}/{task}/{name}/{variant}';predictions[k]=pred;results[k]=metric(pred,y,metric_scale,ep[rr],endpoint)
 np.savez(O/'predictions.npz',**predictions);c.write(O/'metric-scales.json',scales_record)
 indices=np.random.default_rng(20260922).integers(0,20,(2000,3,20));paired={}
 for endpoint in ['state','goal']:
  paired[endpoint]={}
  for method in ['ridge','mlp']:
   paired[endpoint][method]={}
   for variant in CONDS:
    a=np.stack([np.mean([results[f'{endpoint}/{method}/{t}/goal_auxiliary_svae{s}/{variant}']['per_episode_groups'] for s in [42,43,44]],0) for t in c.TASKS]);b=np.stack([np.mean([results[f'{endpoint}/{method}/{t}/reconstruction_only_svae{s}/{variant}']['per_episode_groups'] for s in [42,43,44]],0) for t in c.TASKS]);delta=a-b
    ds=np.stack([delta[t,indices[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));bs=np.stack([b[t,indices[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));assert np.all(bs>0)
    paired[endpoint][method][variant]={'E_relative_change_percent':float(100*delta.mean()/b.mean()),'relative_percent95':np.quantile(100*ds/bs,[.025,.975]).tolist(),'baseline_E':float(b.mean()),'auxiliary_E':float(a.mean()),'per_task_relative_percent':(100*delta.mean((1,2))/b.mean((1,2))).tolist(),'per_task_group_relative_percent':(100*delta.mean(1)/b.mean(1)).tolist(),'task_order':c.TASKS,'independent_episodes':60}
 goal=paired['goal']['ridge'];st=paired['state']['ridge'];criteria={'noisy_goal_point_at_least_10pct':goal['noise0.10']['E_relative_change_percent']<=-10,'noisy_goal_upper_below_zero':goal['noise0.10']['relative_percent95'][1]<0,'noisy_state_upper_below_5pct':st['noise0.10']['relative_percent95'][1]<5,'clean_state_upper_below_5pct':st['clean']['relative_percent95'][1]<5,'clean_goal_upper_below_5pct':goal['clean']['relative_percent95'][1]<5,'no_task_noisy_goal_regression_over_10pct':max(goal['noise0.10']['per_task_relative_percent'])<=10,'no_task_noisy_state_regression_over_10pct':max(st['noise0.10']['per_task_relative_percent'])<=10}
 summary={'completed_at':datetime.datetime.now().astimezone().isoformat(),'scope':'independent prospective within-task offline representation readout; no policy benefit measured','episodes':60,'frames':720,'conditions':results,'paired':paired,'joint_primary_criteria':criteria,'joint_primary_success':all(criteria.values()),'refits':0,'independent_unit':'episode; seeds/frames/noise copies averaged within episode','limitations':['three fixed tasks, no unseen-task evidence','waypoint from gripper crossing is not object pose or contact','not a comprehensive RAE-versus-VAE comparison','no closed-loop policy evaluation','one readout MLP seed','initial failed setup repaired without changing collection deadline; all evidence retained'],'sha256':{n:c.sha(O/n) for n in ['predictions.npz','metric-scales.json','labels.json','identity.json','parity.json','freeze.json']}}
 c.write(O/'summary.json',summary);print(json.dumps({'paired':paired,'joint_primary_success':summary['joint_primary_success'],'criteria':criteria}),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='extract':extract()
 elif sys.argv[1]=='score':score()
 else:raise ValueError(sys.argv[1])
