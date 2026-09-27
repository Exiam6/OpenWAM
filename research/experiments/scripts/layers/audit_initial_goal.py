"""Independent raw-event, saved-prediction and bootstrap audit for secondary goal probe."""
from pathlib import Path
import datetime,hashlib,json,os
import h5py,numpy as np,torch
R=Path(os.environ['LAYER_RUN']);O=R/'initial-goal';C=R/'confirmation';tasks=['adjust_bottle','handover_block','place_object_basket'];torch.set_num_threads(6)
summary=json.loads((O/'fresh-summary.json').read_text());labels=json.loads((O/'fresh-labels.json').read_text());manifest=json.loads((R/'fresh/manifest-v2.json').read_text());saved={(r['task'],r['seed']):r for r in labels};truth={}
for task in tasks:
 for rec in manifest['accepted'][task]:
  with h5py.File(rec['hdf5'],'r') as f:
   g=np.column_stack([f['endpose/left_gripper'][:],f['endpose/right_gripper'][:]]);event=None
   for frame in range(1,len(g)):
    arms=np.flatnonzero((g[frame-1]>.5)&(g[frame]<=.5))
    if len(arms):event=(frame,int(arms[0]));break
   assert event is not None;frame,arm=event;side=['left','right'][arm];y=np.asarray(f[f'endpose/{side}_endpose'][frame,:2],dtype=np.float32);key=(task,rec['seed']);truth[key]=y.astype(float);a=saved[key];assert a['event_frame']==frame and a['arm']==side and np.array_equal(y,np.array(a['target_xy'],dtype=np.float32))
assert len(truth)==60
training=json.loads((O/'training-samples.json').read_text());old=json.loads((R/'readout/manifest.json').read_text());allowed={(r['task'],r['episode']) for r in old if r['split']=='train'};assert len(training)==105 and all((r['task'],r['episode']) in allowed and r['frame']==0 for r in training)
constants=json.loads((O/'constants.json').read_text());selection=json.loads((O/'selection.json').read_text())
for task in tasks:
 y=np.array([r['target_xy'] for r in training if r['task']==task]);assert len(y)==35;assert np.array_equal(y.mean(0),np.array(constants[task]['mean']));assert np.array_equal(y.std(0).clip(1e-6),np.array(constants[task]['std']))
 expected=sorted(r['episode'] for r in training if r['task']==task)
 for key,dec in selection.items():
  if key.startswith(task+'/'):assert sorted(i for fold in dec['folds'] for i in fold)==expected
rows=json.loads((C/'sample-manifest.json').read_text());d=np.load(C/'vectors.npz');task=np.array([r['task'] for r in rows]);ep=np.array([r['seed'] for r in rows]);frame=np.array([r['frame'] for r in rows]);ref=d['reference'];cond=d['condition'];pred=np.load(O/'fresh-predictions.npz');coef=np.load(O/'coefficients.npz');models=torch.load(O/'models.pt',map_location='cpu',weights_only=False);computed={};maxdiff=0.
for key,p in pred.items():
 method,t,name,variant=key.split('/');ii=np.flatnonzero((task[ref]==t)&(frame[ref]==0)&(cond==variant));rr=ref[ii];y=np.array([truth[t,ep[i]] for i in rr]);x=d[name][ii]
 if method=='ridge':
  c={k:coef[f'{t}/{name}/{k}'] for k in ['xm','xs','ym','ys','w']};q=((x-c['xm'])/c['xs']/np.sqrt(x.shape[1]))@c['w']*c['ys']+c['ym']
 else:
  xm,xs,ym,ys=[v.numpy() for v in models['scales'][t+'/'+name]];z=(x-xm)/xs;ww=models['states'][t+'/'+name]
  for layer in [0,2,4]:
   z=z@ww[f'{layer}.weight'].numpy().T+ww[f'{layer}.bias'].numpy()
   if layer<4:z=np.maximum(z,0)
  q=z*ys+ym
 assert np.allclose(p,q,rtol=1e-5,atol=1e-6);maxdiff=max(maxdiff,float(np.max(np.abs(p-q))));raw=(p-y)**2;error=raw/np.array(constants[t]['std'])**2;unique=np.unique(ep[rr]);per=np.array([error[ep[rr]==j].mean(0) for j in unique]);s=summary['conditions'][key]
 assert unique.tolist()==s['episode_ids'];assert np.allclose(per,s['per_episode_xy_nmse'],rtol=1e-10,atol=1e-12);assert np.isclose(per.mean(),s['E'],rtol=1e-10);assert np.isclose(np.sqrt(raw.sum(1).mean()),s['planar_distance_RMSE_native_units'],rtol=1e-10);computed[key]=per
ix=np.random.default_rng(20260922).integers(0,20,(2000,3,20));checks={}
for method in ['ridge','mlp']:
 for variant in ['clean','noise0.10','noise0.04']:
  a=np.stack([np.mean([computed[f'{method}/{t}/K4_svae{s}/{variant}'] for s in [42,43,44]],axis=0) for t in tasks]);b=np.stack([np.mean([computed[f'{method}/{t}/L12_svae{s}/{variant}'] for s in [42,43,44]],axis=0) for t in tasks]);delta=a-b;boot=[]
  for z in ix:boot.append(100*np.mean([delta[t,z[t]].mean() for t in range(3)])/np.mean([b[t,z[t]].mean() for t in range(3)]))
  ci=np.quantile(boot,[.025,.975]);s=summary['secondary_K4_SVAE_vs_L12_SVAE'][method][variant];assert np.allclose(ci,s['relative_percent95'],rtol=1e-10,atol=1e-12);assert np.isclose(100*delta.mean()/b.mean(),s['E_relative_change_percent'],rtol=1e-10);checks[method+'/'+variant]=ci.tolist()
for t in tasks:
 yy=np.array([truth[t,rec['seed']] for rec in manifest['accepted'][t]]);base=((yy-np.array(constants[t]['mean']))/np.array(constants[t]['std']))**2;assert np.isclose(base.mean(),summary['conditions']['constant/'+t]['E'],rtol=1e-12)
result={'passed':True,'checked_at':datetime.datetime.now().astimezone().isoformat(),'raw_hdf5_event_labels_recomputed':60,'train_only_initial_samples_verified':105,'prediction_conditions_recomputed':len(computed),'prediction_forward_max_abs':maxdiff,'paired_intervals_recomputed':checks,'constant_baselines_recomputed':3,'primary_protocol_unchanged':True,'scope':'secondary probe audit, not a new independent replication or policy benefit'};(O/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
