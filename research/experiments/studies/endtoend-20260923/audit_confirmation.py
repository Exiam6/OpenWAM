"""Independent saved-artifact audit. No fitting, encoder runs, or cohort retries."""
import datetime,hashlib,json,os
from pathlib import Path
import h5py,numpy as np
P=Path('/home/zifanz4/openwam-experiments');R=Path('/data02/zifanz4/openwam-experiments');O=R/'results/confirmation-20260923';L=R/'results/layers-20260922';TASKS=['adjust_bottle','handover_block','place_object_basket'];WN=['Wan48_native_scale','Wan48_per_token_layernorm'];NAMES=WN+['L12_raw','L12_pca','L12_svae42','L12_svae43','L12_svae44'];CONDS=['clean','noise0.10','noise0.04']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def main():
 assert not (O/'audit.json').exists() and not (P/'studies/endtoend-20260923/confirmation-audit.json').exists(),'Never repeat completed audit'
 assert not (P/'studies/confirmation-20260923/paused.json').exists()
 assert not Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
 deadline=datetime.datetime.fromisoformat(json.loads((P/'studies/endtoend-20260923/audit-window.json').read_text())['deadline']);assert datetime.datetime.now().astimezone()<deadline
 assert not (P/'studies/endtoend-20260923/paused.json').exists()
 summary=json.loads((O/'result.json').read_text());saved=np.load(O/'predictions.npz');assert json.loads((O/'features/exit.json').read_text())['exit_code']==0
 assert json.loads((O/'features/hardware-parity.json').read_text())['passed'];freeze=json.loads((O/'freeze.json').read_text())
 for path,h in freeze['files'].items():assert sha(path)==h,path
 labels={};states=[];targets=[];rows=[];refs=[];conds=[];arrays={n:[] for n in NAMES};counts={}
 for task in TASKS:
  d=O/'fresh'/task;manifest=json.loads((d/'manifest.json').read_text());assert manifest['complete'];assert json.loads((d/'exit.json').read_text())['exit_code']==0
  accepted=manifest['accepted'];assert len(accepted)==20;counts[task]=20
  attempted=manifest['attempts'];start=json.loads((O/'protocol.json').read_text())['new_seed_starts'][task]
  assert [x['seed'] for x in attempted]==list(range(start,start+len(attempted))) and len(attempted)<=40
  assert [x['seed'] for x in attempted if x.get('accepted')]==[x['seed'] for x in accepted]
  for rec in accepted:
   seed=rec['seed'];fd=O/'features'/task/f'seed-{seed}';v=np.load(fd/'vectors.npz');rr=json.loads((fd/'rows.json').read_text());assert sha(rec['hdf5'])==rec['hdf5_sha256'];base=len(rows)
   with h5py.File(rec['hdf5'],'r') as f:
    n=len(f['endpose/left_endpose']);ff=np.unique(np.linspace(0,n-33,12,dtype=int));assert len(ff)==12 and [x['frame'] for x in rr]==ff.tolist();events=[]
    for ai,arm in enumerate(['left','right']):
     grip=f[f'endpose/{arm}_gripper'][:];cross=np.flatnonzero((grip[:-1]>.5)&(grip[1:]<=.5))+1
     if len(cross):events.append((int(cross[0]),ai,arm))
    event,ai,arm=min(events);y=f[f'endpose/{arm}_endpose'][event,:2].astype(np.float32);labels[task,seed]=y
    assert np.array_equal(y,np.array(json.loads((fd/'goal.json').read_text())['target_xy'],dtype=np.float32))
    for j,frame in enumerate(ff):
     lp=f['endpose/left_endpose'][frame];rp=f['endpose/right_endpose'][frame]
     state=np.r_[lp,rp,f['endpose/left_gripper'][frame],f['endpose/right_gripper'][frame]].astype(np.float32)
     target=np.r_[f['endpose/left_endpose'][frame+4,:3]-lp[:3],f['endpose/right_endpose'][frame+4,:3]-rp[:3],f['endpose/left_gripper'][frame+4],f['endpose/right_gripper'][frame+4]].astype(np.float32)
     assert np.array_equal(state,v['state'][j]) and np.array_equal(target,v['target'][j])
     assert hashlib.sha256(bytes(f['observation/head_camera/rgb'][frame])).hexdigest()==rr[j]['head_encoded_sha256']
    assert v['reference'].tolist()==np.repeat(np.arange(12),7).tolist()
    assert v['condition'].tolist()==(['clean']+['noise0.10']*3+['noise0.04']*3)*12
   for name in NAMES:arrays[name].append(v[name]);assert np.isfinite(v[name]).all()
   refs.extend((v['reference']+base).tolist());conds.extend(v['condition'].tolist());rows.extend(rr);states.extend(v['state']);targets.extend(v['target'])
 assert rows==json.loads((O/'sample-manifest.json').read_text()) and len(rows)==720
 arrays={k:np.concatenate(v) for k,v in arrays.items()};state=np.array(states,dtype=float);target=np.array(targets,dtype=float);ref=np.array(refs);cond=np.array(conds);tasks=np.array([r['task'] for r in rows]);ep=np.array([r['seed'] for r in rows]);frame=np.array([r['frame'] for r in rows]);goalcoef=np.load(R/'results/rae-noise-control-20260922/coefficients.npz');statecoef=np.load(R/'results/rae-state-control-20260923/coefficients.npz');proprio=np.load(L/'readout/coefficients.npz');constants=json.loads((L/'initial-goal/constants.json').read_text());count=0;maxpred=0.;maxmetric=0.;results={}
 for kind in ['goal','state']:
  results[kind]={}
  for name in NAMES+(['proprio'] if kind=='state' else []):
   for task in TASKS:
    keys=[f'{name}/{task}'] if kind=='goal' else [f'{task}/proprio/{j}' if name=='proprio' else f'{name}/{task}/{j}' for j in [0,1]];coef=goalcoef if kind=='goal' else (proprio if name=='proprio' else statecoef)
    for con in CONDS:
     ii=np.flatnonzero((tasks[ref]==task)&(cond==con)&((frame[ref]==0) if kind=='goal' else True));rr=ref[ii];x=state[rr] if name=='proprio' else arrays[name][ii].astype(float)
     if kind=='state' and name!='proprio':x=np.column_stack((x,state[rr]))
     parts=[]
     for key in keys:
      xm,xs,ym,ys,w=[coef[key+'/'+k] for k in ['xm','xs','ym','ys','w']]
      affine=w*ys[None,:]/xs[:,None]/np.sqrt(len(xm));bias=ym-xm@affine;parts.append(x@affine+bias)
     pred=np.column_stack(parts);key=name+'/'+task+'/'+con;maxpred=max(maxpred,float(abs(pred-saved[kind+'/'+key]).max()));assert np.allclose(pred,saved[kind+'/'+key],rtol=1e-9,atol=1e-10)
     y=np.array([labels[task,int(e)] for e in ep[rr]],dtype=float) if kind=='goal' else target[rr]
     scale=np.array(constants[task]['std']) if kind=='goal' else np.r_[proprio[f'{task}/L12_raw/0/ys'],proprio[f'{task}/L12_raw/1/ys']]
     e=((pred-y)/scale)**2;ids=np.unique(ep[rr]);assert len(ids)==20
     per=np.array([e[ep[rr]==j].mean(0) if kind=='goal' else [e[ep[rr]==j,:6].mean(),e[ep[rr]==j,6:].mean()] for j in ids])
     field='per_episode_xy_nmse' if kind=='goal' else 'per_episode_groups';stored=summary['conditions'][kind][key];assert ids.tolist()==stored['episode_ids'];maxmetric=max(maxmetric,float(abs(per-np.array(stored[field])).max()));assert np.allclose(per,stored[field],rtol=1e-9,atol=1e-10);assert abs(float(per.mean())-stored['E'])<1e-9;results[kind][key]=per;count+=1
 groups={WN[0]:[WN[0]],WN[1]:[WN[1]],'DINO_raw768':['L12_raw'],'DINO_PCA48':['L12_pca'],'DINO_SVAE48_seed_mean':['L12_svae42','L12_svae43','L12_svae44']};ix=np.random.default_rng(20260922).integers(0,20,(2000,3,20));paired_count=0;primary={}
 for kind in results:
  gg=dict(groups)
  if kind=='state':gg['proprio']=['proprio']
  for con in CONDS:
   a={n:np.array([np.mean([results[kind][v+'/'+t+'/'+con] for v in names],0) for t in TASKS]) for n,names in gg.items()}
   for base in WN+(['proprio'] if kind=='state' else []):
    for n in ['DINO_raw768','DINO_PCA48','DINO_SVAE48_seed_mean']:
     for component,sl in [('all',slice(None))]+([('translation',slice(0,1)),('gripper',slice(1,2))] if kind=='state' else []):
      aa=a[n][:,:,sl];bb=a[base][:,:,sl];d=aa-bb;boots=np.array([[d[t,z[t]].mean(),bb[t,z[t]].mean()] for z in ix for t in range(3)]).reshape(2000,3,2).mean(1)
      ci=np.quantile(100*boots[:,0]/boots[:,1],[.025,.975]);ac=np.quantile(boots[:,0],[.025,.975]);stored=summary['paired'][kind][con][base][n];stored=stored if component=='all' else stored[component]
      assert np.allclose(ci,stored['paired95'],atol=1e-7,rtol=1e-9);assert np.allclose(ac,stored['absolute95'],atol=1e-9,rtol=1e-9);paired_count+=1
      if base==WN[0] and n=='DINO_PCA48' and con=='noise0.10' and component=='all':primary[kind]=bool(ac[1]<0)
 assert summary['joint_primary_passed']==all(primary.values());assert count==135
 report={'time':datetime.datetime.now().astimezone().isoformat(),'passed':True,'raw_episodes_audited':60,'raw_state_frames_audited':720,'prediction_conditions':count,'paired_component_intervals':paired_count,'max_expanded_affine_prediction_difference':maxpred,'max_scene_metric_difference':maxmetric,'co_primary_decisions':primary,'joint_primary_passed':all(primary.values()),'no_refits_or_inference':True,'summary_sha256':sha(O/'result.json'),'audit_script_sha256':sha(Path(__file__))}
 (P/'studies/endtoend-20260923/confirmation-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
