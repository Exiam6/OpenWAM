"""Once-only secondary initial-image expert waypoint score; no refitting."""
import datetime,hashlib,json,os
from pathlib import Path
import h5py,numpy as np,torch
R=Path(os.environ['LAYER_RUN']);O=R/'initial-goal';C=R/'confirmation';tasks=['adjust_bottle','handover_block','place_object_basket'];torch.set_num_threads(6)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not (O/'fresh-summary.json').exists();assert json.loads((C/'audit.json').read_text())['passed'];complete=json.loads((O/'complete.json').read_text());start=json.loads((R/'confirmation-started.json').read_text());assert datetime.datetime.fromisoformat(complete['completed_at'])<datetime.datetime.fromisoformat(start['started_at'])
for file,h in complete['sha256'].items():assert sha(O/file)==h,file
fresh=json.loads((R/'fresh/manifest-v2.json').read_text());rows=json.loads((C/'sample-manifest.json').read_text());d=np.load(C/'vectors.npz');ref=d['reference'];cond=d['condition'];task=np.array([r['task'] for r in rows]);ep=np.array([r['seed'] for r in rows]);frame=np.array([r['frame'] for r in rows]);lookup={};labels=[];initial_proprio={};missing=[]
for t in tasks:
 for rec in fresh['accepted'][t]:
  with h5py.File(rec['hdf5'],'r') as f:
   candidates=[]
   for ai,arm in enumerate(['left','right']):
    grip=f[f'endpose/{arm}_gripper'][:];ii=np.flatnonzero((grip[:-1]>.5)&(grip[1:]<=.5))+1
    if len(ii):candidates.append((int(ii[0]),ai,arm))
   if not candidates:missing.append({'task':t,'seed':rec['seed']});continue
   when,ai,arm=min(candidates);y=np.asarray(f[f'endpose/{arm}_endpose'][when,:2],dtype=np.float32);lookup[t,rec['seed']]=y;labels.append({'task':t,'seed':rec['seed'],'event_frame':when,'arm':arm,'target_xy':y.tolist(),'simultaneous_crossing':len(candidates)>1 and candidates[0][0]==candidates[1][0],'hdf5_sha256':rec['hdf5_sha256']})
if missing:
 (O/'incomplete-labels.json').write_text(json.dumps({'missing':missing,'replacement_allowed':False},indent=2)+'\n');raise RuntimeError('Missing planned event labels; preserve and do not silently change sample set')
assert len(labels)==60
for t in tasks:
 xx=d['state'][(task==t)&(frame==0)];assert len(xx)==20;initial_proprio[t]={'range':np.ptp(xx,axis=0).tolist(),'constant_within_task':bool(np.all(np.ptp(xx,axis=0)==0))}
coef=np.load(O/'coefficients.npz');models=torch.load(O/'models.pt',map_location='cpu',weights_only=False);constants=json.loads((O/'constants.json').read_text());names=[s+'_'+v for s in ['L12','L6','K4'] for v in ['raw','pca','svae42','svae43','svae44']];results={};preds={};model_check=0.
def metric(p,y,scale,episode):
 e=(p-y)**2;err=e/scale**2;ee=np.unique(episode);per=np.array([err[episode==j].mean(axis=0) for j in ee]);raw=np.array([e[episode==j].mean(axis=0) for j in ee]);return {'episode_ids':ee.tolist(),'per_episode_xy_nmse':per.tolist(),'E':float(per.mean()),'planar_distance_RMSE_native_units':float(np.sqrt(raw.sum(axis=1).mean()))}
for t in tasks:
 scale=np.array(constants[t]['std']);mu=np.array(constants[t]['mean']);yy=np.array([lookup[t,r['seed']] for r in rows if r['task']==t and r['frame']==0],dtype=float);ee=ep[(task==t)&(frame==0)];results['constant/'+t]=metric(np.broadcast_to(mu,yy.shape),yy,scale,ee)
for method in ['ridge','mlp']:
 for t in tasks:
  for name in names:
   key=t+'/'+name
   if method=='ridge':m={k:coef[key+'/'+k] for k in ['xm','xs','ym','ys','w']}
   else:
    xm,xs,ym,ys=[v.numpy() for v in models['scales'][key]];model=torch.nn.Sequential(torch.nn.Linear(len(xm),128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,2));model.load_state_dict(models['states'][key]);model.eval()
   for variant in ['clean','noise0.10','noise0.04']:
    ii=np.flatnonzero((task[ref]==t)&(frame[ref]==0)&(cond==variant));rr=ref[ii];x=d[name][ii];y=np.array([lookup[t,ep[j]] for j in rr],dtype=float)
    if method=='ridge':p=((x-m['xm'])/m['xs']/np.sqrt(x.shape[1]))@m['w']*m['ys']+m['ym']
    else:
     with torch.inference_mode():p=model(torch.from_numpy((x-xm)/xs)).numpy()*ys+ym
     z=(x-xm)/xs;ww=models['states'][key]
     for layer in [0,2,4]:
      z=z@ww[f'{layer}.weight'].numpy().T+ww[f'{layer}.bias'].numpy()
      if layer<4:z=np.maximum(z,0)
     q=z*ys+ym;model_check=max(model_check,float(np.max(np.abs(p-q))));assert np.allclose(p,q,rtol=1e-5,atol=1e-6),(key,variant)
    k=method+'/'+key+'/'+variant;results[k]=metric(p,y,np.array(constants[t]['std']),ep[rr]);preds[k]=p
paired={};indices=np.random.default_rng(20260922).integers(0,20,(2000,3,20));matrix={}
for method in ['ridge','mlp']:
 paired[method]={};matrix[method]={}
 for variant in ['clean','noise0.10','noise0.04']:
  a=np.stack([np.mean([results[f'{method}/{t}/K4_svae{s}/{variant}']['per_episode_xy_nmse'] for s in [42,43,44]],axis=0) for t in tasks]);b=np.stack([np.mean([results[f'{method}/{t}/L12_svae{s}/{variant}']['per_episode_xy_nmse'] for s in [42,43,44]],axis=0) for t in tasks]);delta=a-b;ds=np.stack([delta[t,indices[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));bs=np.stack([b[t,indices[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));paired[method][variant]={'E_relative_change_percent':float(100*delta.mean()/b.mean()),'relative_percent95':np.quantile(100*ds/bs,[.025,.975]).tolist(),'per_task_relative_percent':(100*delta.mean(axis=(1,2))/b.mean(axis=(1,2))).tolist(),'task_order':tasks}
 for source in ['L12','L6','K4']:
  for comp in ['raw','pca','svae']:
   names_here=[source+'_'+comp+str(s) for s in [42,43,44]] if comp=='svae' else [source+'_'+comp];matrix[method][source+'_'+comp]={v:float(np.mean([results[f'{method}/{t}/{n}/{v}']['E'] for t in tasks for n in names_here])) for v in ['clean','noise0.10','noise0.04']}
np.savez(O/'fresh-predictions.npz',**preds);(O/'fresh-labels.json').write_text(json.dumps(labels,indent=2)+'\n');summary={'scope':'prospectively fixed secondary initial-scene expert waypoint readout, added after head-development outcomes; not primary rescue or policy improvement','episodes':60,'initial_proprio':initial_proprio,'constant_baseline_E':float(np.mean([results['constant/'+t]['E'] for t in tasks])),'conditions':results,'all_source_compressor_E':matrix,'secondary_K4_SVAE_vs_L12_SVAE':paired,'fresh_refits':0,'missing_events':0,'simultaneous_crossing_count':sum(x['simultaneous_crossing'] for x in labels),'independent_numpy_mlp_max_abs':model_check,'training_size_per_task':35,'limits':['firstclose waypoint, not object pose or physical contact','same60episodes as primary, not another independent replication','one readout seed and3fixed reducer seeds','3fixed tasks, no unseen-task generalization']};(O/'fresh-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'secondary':paired,'matrix':matrix,'constant_E':summary['constant_baseline_E']},indent=2))
