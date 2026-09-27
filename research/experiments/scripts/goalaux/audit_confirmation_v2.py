"""Independent CPU verification of saved predictions, raw labels and intervals."""
import json,os,datetime,hashlib
from pathlib import Path
import h5py,numpy as np,torch
R=Path(os.environ['WAM_ROOT']);B=R/'results/goalaux-20260922';L=R/'results/layers-20260922';O=B/'confirmation-v2';torch.set_num_threads(4)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'audit.json').exists();summary=json.loads((O/'summary.json').read_text())
for f,h in summary['sha256'].items():assert sha(O/f)==h,f
rows=json.loads((O/'sample-manifest.json').read_text());v=np.load(O/'vectors.npz');pred=np.load(O/'predictions.npz');scales=json.loads((O/'metric-scales.json').read_text());manifest=json.loads((B/'fresh-v3/manifest.json').read_text());labels=json.loads((O/'labels.json').read_text());assert len(rows)==720 and len(pred.files)==432 and len(labels['records'])==60
lookup={(r['task'],r['seed']):r for t in manifest['accepted'].values() for r in t};goal={};raw_label_checks=0
for (task,seed),rec in lookup.items():
 idx=[i for i,r in enumerate(rows) if r['task']==task and r['seed']==seed];assert len(idx)==12
 with h5py.File(rec['hdf5'],'r') as f:
  starts=np.linspace(0,len(f['endpose/left_endpose'])-33,12,dtype=int);assert [rows[i]['frame'] for i in idx]==starts.tolist();events=[]
  for ai,arm in enumerate(['left','right']):
   grip=f['endpose/'+arm+'_gripper'][:];crossings=[j for j in range(1,len(grip)) if grip[j-1]>.5 and grip[j]<=.5]
   if crossings:events.append((crossings[0],ai,arm))
  event,_,arm=sorted(events)[0];goal[task,seed]=f['endpose/'+arm+'_endpose'][event,:2].astype(np.float32);lr=next(q for q in labels['records'] if q['task']==task and q['seed']==seed);assert lr['event_frame']==event and lr['arm']==arm and np.array_equal(goal[task,seed],lr['target_xy'])
  for i,st in zip(idx,starts):
   lp=f['endpose/left_endpose'][st];rp=f['endpose/right_endpose'][st];s=np.r_[lp,rp,f['endpose/left_gripper'][st],f['endpose/right_gripper'][st]].astype(np.float32);y=np.r_[f['endpose/left_endpose'][st+4,:3]-lp[:3],f['endpose/right_endpose'][st+4,:3]-rp[:3],f['endpose/left_gripper'][st+4],f['endpose/right_gripper'][st+4]].astype(np.float32)
   assert np.array_equal(v['state'][i],s) and np.array_equal(v['target'][i],y);raw_label_checks+=1
coeff={'new':np.load(B/'readouts/coefficients.npz'),'state':np.load(L/'readout/coefficients.npz'),'goal':np.load(L/'initial-goal/coefficients.npz')};models={'new':torch.load(B/'readouts/mlp.pt',map_location='cpu',weights_only=False),'state':torch.load(L/'readout/mlp/models.pt',map_location='cpu',weights_only=False),'goal':torch.load(L/'initial-goal/models.pt',map_location='cpu',weights_only=False)};constants=json.loads((L/'initial-goal/constants.json').read_text());taskvec=np.array([r['task'] for r in rows]);ep=np.array([r['seed'] for r in rows]);fr=np.array([r['frame'] for r in rows]);ref=v['reference'];cond=v['condition'];recomputed={};max_pred=0
for key in pred.files:
 endpoint,method,task,name,variant=key.split('/');mask=(taskvec[ref]==task)&(cond==variant)
 if endpoint=='goal':mask&=fr[ref]==0
 ii=np.where(mask)[0];rr=ref[ii];y=v['target'][rr].astype(float) if endpoint=='state' else np.array([goal[task,int(ep[j])] for j in rr],dtype=float);scale=np.array(scales['/'.join(key.split('/')[:-1])]);saved_pred=pred[key];assert saved_pred.shape==y.shape and np.isfinite(saved_pred).all()
 if name=='constant':manual=np.tile(constants[task]['mean'],(len(y),1))
 else:
  x=v['state'][rr] if name=='proprio' else (np.c_[v[name][ii],v['state'][rr]] if endpoint=='state' else v[name][ii]);kind='new' if name.startswith(('reconstruction_only','goal_auxiliary')) else endpoint;modelkey=f'{endpoint}/{task}/{name}' if kind=='new' else f'{task}/{name}'
  if method=='ridge':
   out=[]
   for g in ([0,1] if endpoint=='state' else [0]):
    prefix=modelkey+(f'/{g}' if endpoint=='state' or kind=='new' else '');a={f:coeff[kind][prefix+'/'+f] for f in ['xm','xs','ym','ys','w']};out.append(np.matmul((x.astype(float)-a['xm'])/a['xs'],a['w']/np.sqrt(x.shape[1]))*a['ys']+a['ym'])
   manual=np.concatenate(out,axis=1)
  else:
   a=models[kind];xm,xs,ym,ys=[q.numpy() for q in a['scales'][modelkey]];z=(x-xm)/xs;weights=a['states'][modelkey]
   for layer in [0,2,4]:
    z=np.matmul(z,weights[f'{layer}.weight'].numpy().T)+weights[f'{layer}.bias'].numpy()
    if layer<4:z=np.maximum(z,0)
   manual=z*ys+ym
 assert np.allclose(saved_pred,manual,rtol=3e-5,atol=3e-6),key;max_pred=max(max_pred,float(np.max(np.abs(saved_pred-manual))))
 errors=np.square((saved_pred.astype(float)-y)/scale);unique=sorted(set(ep[rr]));groups=[]
 for seed in unique:
  er=errors[ep[rr]==seed];groups.append([float(np.mean(er[:,:6])),float(np.mean(er[:,6:]))] if endpoint=='state' else np.mean(er,axis=0).tolist())
 groups=np.array(groups);reported=summary['conditions'][key];assert unique==reported['episode_ids'];assert np.allclose(groups,reported['per_episode_groups'],rtol=1e-12,atol=1e-12) and np.isclose(groups.mean(),reported['E'],rtol=1e-12,atol=1e-12);recomputed[key]=groups
T=['adjust_bottle','handover_block','place_object_basket'];ix=np.random.default_rng(20260922).integers(0,20,(2000,3,20));interval_checks=0
for endpoint in ['state','goal']:
 for method in ['ridge','mlp']:
  for condition in ['clean','noise0.10','noise0.04']:
   a=np.array([sum(recomputed[f'{endpoint}/{method}/{t}/goal_auxiliary_svae{s}/{condition}'] for s in [42,43,44])/3 for t in T]);b=np.array([sum(recomputed[f'{endpoint}/{method}/{t}/reconstruction_only_svae{s}/{condition}'] for s in [42,43,44])/3 for t in T]);boot=[]
   for draw in ix:
    am=np.mean([np.mean(a[i,draw[i]]) for i in range(3)]);bm=np.mean([np.mean(b[i,draw[i]]) for i in range(3)]);boot.append(100*(am-bm)/bm)
   p=summary['paired'][endpoint][method][condition];assert np.allclose(np.quantile(boot,[.025,.975]),p['relative_percent95'],rtol=1e-10,atol=1e-10);assert np.isclose(100*(a.mean()-b.mean())/b.mean(),p['E_relative_change_percent'],rtol=1e-10,atol=1e-10);interval_checks+=1
p=summary['paired'];g=p['goal']['ridge'];s=p['state']['ridge'];decision=(g['noise0.10']['E_relative_change_percent']<=-10 and g['noise0.10']['relative_percent95'][1]<0 and s['noise0.10']['relative_percent95'][1]<5 and s['clean']['relative_percent95'][1]<5 and g['clean']['relative_percent95'][1]<5 and all(x<=10 for x in g['noise0.10']['per_task_relative_percent']+s['noise0.10']['per_task_relative_percent']));assert decision==summary['joint_primary_success']==all(summary['joint_primary_criteria'].values())
out={'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),'raw_state_labels_checked':raw_label_checks,'raw_goal_events_checked':60,'predictions_and_metrics_checked':len(pred.files),'paired_intervals_checked':interval_checks,'max_prediction_numpy_difference':max_pred,'joint_primary_decision_verified':True,'joint_primary_success':decision,'summary_sha256':sha(O/'summary.json'),'no_refit_or_scene_replay':True}
(O/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
