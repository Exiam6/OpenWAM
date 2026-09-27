"""Secondary same-camera baseline; score frozen wrist readouts, never refit."""
import datetime,hashlib,json,os
from pathlib import Path
import numpy as np,torch
R=Path(os.environ['LAYER_RUN']);O=R/'wrist-baseline';C=R/'confirmation';tasks=['adjust_bottle','handover_block','place_object_basket'];torch.set_num_threads(6)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not (O/'fresh-summary.json').exists();assert json.loads((C/'audit.json').read_text())['passed'];complete=json.loads((O/'complete.json').read_text());started=json.loads((R/'confirmation-started.json').read_text());assert datetime.datetime.fromisoformat(complete['completed_at'])<datetime.datetime.fromisoformat(started['started_at']),'Baseline weights must precede independent confirmation execution'
for name,h in complete['sha256'].items():assert sha(O/name)==h,name
rows=json.loads((C/'sample-manifest.json').read_text());d=np.load(C/'vectors.npz');ref=d['reference'];condition=d['condition'];task=np.array([r['task'] for r in rows]);ep=np.array([r['seed'] for r in rows]);coef=np.load(O/'coefficients.npz');mlp=torch.load(O/'models.pt',map_location='cpu',weights_only=False);original=json.loads((C/'summary.json').read_text());names=[s+'_'+v for s in ['L12','L6','K4'] for v in ['raw','pca','svae42','svae43','svae44']];out={};predictions={};largest_numpy_mlp_difference=0.
for method in ['ridge','mlp']:
 for t in tasks:
  ii=np.flatnonzero((task[ref]==t)&(condition=='left_transfer'));rr=ref[ii];assert len(rr)==240
  for name in names:
   key=t+'/'+name;x=np.concatenate([d[name][ii],d['state'][rr]],axis=1);y=d['target'][rr]
   if method=='ridge':
    pp=[];ss=[]
    for gi in [0,1]:
     m={k:coef[f'{key}/{gi}/{k}'] for k in ['xm','xs','ym','ys','w']};pp.append(((x-m['xm'])/m['xs']/np.sqrt(x.shape[1]))@m['w']*m['ys']+m['ym']);ss.append(m['ys'])
    p=np.concatenate(pp,axis=1);scale=np.concatenate(ss);y=y.astype(float)
   else:
    xm,xs,ym,scale=[v.numpy() for v in mlp['scales'][key]];m=torch.nn.Sequential(torch.nn.Linear(len(xm),128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,8));m.load_state_dict(mlp['states'][key]);m.eval()
    with torch.inference_mode():p=m(torch.from_numpy((x-xm)/xs)).numpy()*scale+ym
    z=(x-xm)/xs;weights=mlp['states'][key]
    for layer in [0,2,4]:
     z=z@weights[f'{layer}.weight'].numpy().T+weights[f'{layer}.bias'].numpy()
     if layer<4:z=np.maximum(z,0)
    q=z*scale+ym;largest_numpy_mlp_difference=max(largest_numpy_mlp_difference,float(np.max(np.abs(p-q))));assert np.allclose(p,q,rtol=1e-5,atol=1e-6),(method,key)
   error=((p-y)/scale)**2;ee=np.unique(ep[rr]);per=np.array([[error[ep[rr]==j,:6].mean(),error[ep[rr]==j,6:].mean()] for j in ee]);assert len(ee)==20;assert np.isfinite(per).all();out[method+'/'+key]={'episode_ids':ee.tolist(),'per_episode_groups':per.tolist(),'E':float(per.mean()),'translation':float(per[:,0].mean()),'gripper':float(per[:,1].mean())};predictions[method+'/'+key]=p
comparison={}
for method in ['ridge','mlp']:
 comparison[method]={}
 for source in ['L12','L6','K4']:
  for comp in ['raw','pca','svae']:
   nn=[source+'_'+comp+str(s) for s in [42,43,44]] if comp=='svae' else [source+'_'+comp]
   record={}
   for case,values in [('head_train_head_input',[original['conditions'][method][f'{t}/{n}/clean'] for t in tasks for n in nn]),('head_train_wrist_input',[original['conditions'][method][f'{t}/{n}/left_transfer'] for t in tasks for n in nn]),('wrist_train_wrist_input',[out[f'{method}/{t}/{n}'] for t in tasks for n in nn])]:record[case]={m:float(np.mean([v[m] for v in values])) for m in ['E','translation','gripper']}
   comparison[method][source+'_'+comp]=record
p=O/'fresh-predictions.npz';np.savez(p,**predictions)
summary={'scope':'secondary cross-camera control; shared head-trained encoders/compressors, only readout training camera differs','fresh_episodes':60,'fresh_refits':0,'baseline_frozen_before_confirmation':True,'all90visual_conditions':out,'task_balanced_camera_comparison':comparison,'independent_numpy_mlp_forward_max_abs':largest_numpy_mlp_difference,'limitations':['not pure geometric viewpoint invariance','reducers remain head-trained','same fixed tasks and one readout training seed','secondary analysis, cannot replace failed primary noise comparison'],'prediction_sha256':sha(p)}
(O/'fresh-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'conditions':len(out),'numpy_mlp_max_abs':largest_numpy_mlp_difference,'camera_comparison':comparison},indent=2))
