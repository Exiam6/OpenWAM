"""Equal-noise short-state control, cached initial-frame reuse, equivalent ridge."""
import gc,hashlib,json,sys,time,datetime,zipfile
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
import rae_noise_control as g
R=g.R;L=g.L;B=g.B;G=g.O;O=R/'results/rae-state-control-20260923';NAMES=g.NAMES;TASKS=g.TASKS;CONDS=g.CONDS;ALPHAS=[1e-4,.01,1.,100.]
def write(n,d):
 p=O/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def spectral_cache(x,y):
 xm=x.mean(0);xs=x.std(0).clip(1e-6);ym=y.mean(0);ys=y.std(0).clip(1e-6);z=(x-xm)/xs/np.sqrt(x.shape[1]);t=(y-ym)/ys;dual=x.shape[1]>len(x);gram=z@z.T if dual else z.T@z;ev,u=np.linalg.eigh(gram);assert ev.min()>-1e-7*max(1.,float(ev.max()));rhs=t if dual else z.T@t;return (dict(xm=xm,xs=xs,ym=ym,ys=ys),z,ev.clip(0),u,u.T@rhs,dual)
def spectral_fit(cache,alpha):
 m,z,ev,u,rhs,dual=cache;w=u@(rhs/(ev[:,None]+alpha));w=z.T@w if dual else w;return dict(m,w=w)
def check_solver():
 rng=np.random.default_rng(20260923);maximum=0.
 for n,p in [(35,50),(50,20)]:
  x=rng.normal(size=(n,p));y=rng.normal(size=(n,8));xx=rng.normal(size=(9,p));cache=spectral_cache(x,y)
  for alpha in ALPHAS:
   a=g.fit(x,y,7*alpha);b=spectral_fit(cache,7*alpha);maximum=max(maximum,float(abs(g.predict(a,xx)-g.predict(b,xx)).max()))
 assert maximum<1e-9
 return {'passed':True,'max_abs_prediction_difference_vs_direct_solve':maximum,'cases':'wide dual and narrow primal, all4frozen alphas,8outputs','no_actual_readouts_fitted':True}
def select_state(x,y,ep):
 assert all((ep==e).sum()==84 for e in np.unique(ep));folds=np.array_split(np.random.default_rng(20260922).permutation(np.unique(ep)),5);losses=np.zeros((4,5,2))
 for fi,fold in enumerate(folds):
  va=np.isin(ep,fold);cache=spectral_cache(x[~va],y[~va])
  for ai,alpha in enumerate(ALPHAS):
   m=spectral_fit(cache,alpha*7);err=((g.predict(m,x[va])-y[va])/m['ys'])**2
   losses[ai,fi]=np.mean([[err[ep[va]==e,:6].mean(),err[ep[va]==e,6:].mean()] for e in fold],axis=0)
  print('CV_FOLD',fi,flush=True)
 cache=spectral_cache(x,y);models=[];choices=[]
 for gi,sl in enumerate([slice(0,6),slice(6,8)]):
  means=losses[:,:,gi].mean(1);ai=int(np.argmin(means));m=spectral_fit(cache,ALPHAS[ai]*7);models.append(dict(xm=m['xm'],xs=m['xs'],ym=m['ym'][sl],ys=m['ys'][sl],w=m['w'][:,sl]));choices.append({'selected_base_alpha':ALPHAS[ai],'effective_alpha':ALPHAS[ai]*7,'folds':[a.tolist() for a in folds],'all_scores':[{'alpha':alpha,'fold_scores':losses[j,:,gi].tolist(),'score':float(means[j])} for j,alpha in enumerate(ALPHAS)],'variants_per_original_frame':7,'original_frames_per_scene':12})
 return models,choices

def main():
 assert not (O/'result.json').exists();start=time.monotonic();freeze=json.loads((O/'freeze.json').read_text())
 for path,h in freeze['files'].items():assert g.interface.sha(path)==h,path
 assert json.loads((O/'solver-check.json').read_text())['passed'];assert json.loads((G/'hardware-parity.json').read_text())['passed'];torch.set_num_threads(8);torch.manual_seed(20260922);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 rows=json.loads((B/'train-rows.json').read_text());assert rows==json.loads((O/'clean-rows.json').read_text()) and len(rows)==1260
 clean={k:v for k,v in np.load(O/'clean-dino-state-vectors.npz').items()};wv=np.load(B/'wan-training-vectors.npz');clean.update({k:wv[k] for k in g.WN});gv=np.load(G/'training-vectors.npz');grows=json.loads((G/'clean-rows.json').read_text());gl={(x['task'],x['episode']):j for j,x in enumerate(grows)}
 noisy={k:np.empty((7560,clean[k].shape[1]),dtype=np.float32) for k in NAMES};pending=[];reused=[]
 for i,row in enumerate(rows):
  if row['frame']==0:
   gi=gl[row['task'],row['episode']];ids=105+6*gi+np.arange(6);assert np.all(gv['reference'][ids]==gi);assert gv['condition'][ids].tolist()==['noise0.10']*3+['noise0.04']*3
   for k in NAMES:noisy[k][6*i:6*i+6]=gv[k][ids]
   reused.append(i)
  else:pending.append(i)
 assert len(reused)==105 and len(pending)==1155
 weights=torch.load(R/'assets/wan22-vae-baseline/Wan2.2_VAE.pth',map_location='cpu',weights_only=True,mmap=True);weights=weights.get('model_state',weights);model=g.interface.WanVideoVAE38();model.load_state_dict({'model.'+k:v for k,v in weights.items()},strict=True,assign=True);wan=g.interface.WanVideoVAEEncoder(model.to('cuda',dtype=torch.bfloat16)).eval().requires_grad_(False);dino=g.c.encoder();torch.cuda.reset_peak_memory_stats();raw_features=[];new_ids=[]
 with zipfile.ZipFile(B/'train-images.zip') as zf:
  for j,i in enumerate(pending):
   row=rows[i];raw=zf.read(str(i)+'.jpg');assert hashlib.sha256(raw).hexdigest()==row['encoded_head_hashes'][0];arr=np.array(g.c.resize_head(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)))
   for si,sigma in enumerate([.1,.04]):
    for draw in range(3):
     ix=6*i+3*si+draw;rng=np.random.default_rng(20260922+row['episode_uid']*100000+row['frame']*10+draw);im=Image.fromarray(np.clip(arr.astype(np.float32)+rng.normal(0,255*sigma,arr.shape),0,255).astype(np.uint8))
     for k,x in zip(g.WN,g.wan_vectors(wan,im)):noisy[k][ix]=x
     zz=g.c.encode(dino,dino.preprocess_video([im]*9));raw_features.append(zz['L12'][0,:,:1].cpu().half());new_ids.append(ix)
   if j%50==0:write('progress.json',{'stage':'new_noninitial_training_features','completed_original_frames':j+1,'total_original_frames':1155,'elapsed_seconds':time.monotonic()-start});print('ENCODED',j+1,'/1155',flush=True)
 del wan,model,dino;gc.collect();torch.cuda.empty_cache();assert len(new_ids)==6930;features=g.visual(torch.stack(raw_features),'L12');del raw_features;gc.collect()
 for k in g.DN:noisy[k][new_ids]=features[k]
 peak=torch.cuda.max_memory_allocated()/1e9;del features;torch.cuda.empty_cache();ref=np.concatenate([np.arange(1260),np.repeat(np.arange(1260),6)]);condition=np.array(['clean']*1260+(['noise0.10']*3+['noise0.04']*3)*1260);arrays={k:np.concatenate([clean[k],noisy[k]]) for k in NAMES};assert len(ref)==8820 and all(np.isfinite(a).all() for a in arrays.values())
 np.savez(O/'training-vectors.npz',**arrays,reference=ref,condition=condition);write('feature-complete.json',{'completed_at':datetime.datetime.now().astimezone().isoformat(),'new_images':6930,'reused_clean_images':1260,'reused_noisy_initial_images':630,'total_training_rows':8820,'peak_allocated_GB':peak,'new_noisy_original_indices':pending,'reused_noisy_original_indices':reused});del noisy;gc.collect()
 # CPU-only readout stage; no evaluation data or labels accessed yet.
 lab=np.load(B/'train-labels.npz');state=lab['state'].astype(float);y=lab['target'].astype(float);tasks=np.array([x['task'] for x in rows]);eps=np.array([x['episode'] for x in rows]);coef={};choices={};models={}
 for name in NAMES:
  for task in TASKS:
   mask=tasks[ref]==task;x=np.concatenate([arrays[name][mask],state[ref[mask]]],1).astype(float);print('FIT_START',name,task,x.shape,flush=True);mm,cc=select_state(x,y[ref[mask]],eps[ref[mask]]);models[name,task]=mm
   for gi,(m,ch) in enumerate(zip(mm,cc)):
    key=name+'/'+task+'/'+str(gi);choices[key]=ch
    for k,v in m.items():coef[key+'/'+k]=v
   write('progress.json',{'stage':'fit_ridge','completed_group_readouts':len(choices),'total_group_readouts':42,'elapsed_seconds':time.monotonic()-start});print('FIT_END',name,task,flush=True)
 np.savez(O/'coefficients.npz',**coef);write('selection.json',choices);write('fit-complete.json',{'completed_at':datetime.datetime.now().astimezone().isoformat(),'group_fits':42,'eval_vectors_or_labels_accessed':False})
 evd=np.load(L/'confirmation/vectors.npz');evw=np.load(B/'wan-evaluation-vectors.npz');erows=json.loads((L/'confirmation/sample-manifest.json').read_text());etask=np.array([x['task'] for x in erows]);eep=np.array([x['seed'] for x in erows]);estate=evd['state'].astype(float);target=evd['target'].astype(float);originalcoef=np.load(L/'readout/coefficients.npz');results={};preds={}
 from readout_v2 import scores
 for name in NAMES:
  data=evw if name in g.WN else evd;rr=data['reference'];cc=data['condition']
  for task in TASKS:
   mm=models[name,task];scale=np.concatenate([m['ys'] for m in mm]);reference_scale=np.concatenate([originalcoef[f'{task}/L12_raw/{gi}/ys'] for gi in [0,1]]);assert np.allclose(scale,reference_scale,atol=1e-12,rtol=1e-12)
   for con in CONDS:
    ii=np.flatnonzero((etask[rr]==task)&(cc==con));ri=rr[ii];x=np.concatenate([data[name][ii],estate[ri]],1);pred=np.concatenate([g.predict(m,x) for m in mm],1);key=name+'/'+task+'/'+con;results[key]=scores(pred,target[ri],reference_scale,eep[ri]);preds[key]=pred
 groups={g.WN[0]:[g.WN[0]],g.WN[1]:[g.WN[1]],'DINO_raw768':['L12_raw'],'DINO_PCA48':['L12_pca'],'DINO_SVAE48_seed_mean':['L12_svae42','L12_svae43','L12_svae44']};oldw=json.loads((B/'probe-result.json').read_text())['conditions'];oldd=json.loads((L/'confirmation/summary.json').read_text())['conditions']['ridge'];ix=np.random.default_rng(20260922).integers(0,20,(2000,3,20));matrix={};within={};between={}
 def contrast(a,b):
  d=a-b;ds=np.stack([d[t,ix[:,t]] for t in range(3)],1).mean((1,2,3));bs=np.stack([b[t,ix[:,t]] for t in range(3)],1).mean((1,2,3));return {'relative_change_percent':float(100*d.mean()/b.mean()),'descriptive95':np.quantile(100*ds/bs,[.025,.975]).tolist(),'per_task_relative_percent':(100*d.mean((1,2))/b.mean((1,2))).tolist(),'per_task_group_relative_percent':(100*d.mean(1)/b.mean(1)).tolist()}
 for con in CONDS:
  a={};b={}
  for label,names in groups.items():
   aa=[];bb=[]
   for task in TASKS:
    new=[results[n+'/'+task+'/'+con] for n in names];old=[oldw[f'state/{n}/{task}/{con}'] if n in g.WN else oldd[f'{task}/{n}/{con}'] for n in names];assert all(x['episode_ids']==new[0]['episode_ids'] for x in new+old);aa.append(np.mean([x['per_episode_groups'] for x in new],0));bb.append(np.mean([x['per_episode_groups'] for x in old],0))
   a[label]=np.array(aa);b[label]=np.array(bb)
  matrix[con]={n:{'noise_adapted_E':float(a[n].mean()),'original_clean_trained_E':float(b[n].mean()),'noise_adapted_per_task_E':a[n].mean((1,2)).tolist(),'noise_adapted_per_task_groups':a[n].mean(1).tolist()} for n in groups};within[con]={n:contrast(a[n],b[n]) for n in groups};between[con]={base:{n:contrast(a[n],a[base]) for n in ['DINO_raw768','DINO_PCA48','DINO_SVAE48_seed_mean']} for base in g.WN}
 np.savez(O/'predictions.npz',**preds);write('result.json',{'completed_at':datetime.datetime.now().astimezone().isoformat(),'scope':'Exploratory equal-noise short-state control on exposed60; no fullRAE or closedloop claims','training_episodes':105,'evaluation_episodes':60,'new_readout_group_fits':42,'task_order':TASKS,'conditions':results,'matrix':matrix,'within_route_noise_adapted_minus_original':within,'between_noise_adapted_DINO_minus_Wan':between,'elapsed_seconds':time.monotonic()-start,'peak_allocated_GB':peak,'limits':['same exposed60 and known perturbation strengths; no independent confirmation','threeSVAEseeds average errors within same episodes','DINOraw3072visual dimensions versus192for compact controls','approximate cross-device parity inherited from threefixedtrainingframes, not bitidentity','state targets are end-effector deltas/grippers, not contact or controller actions','pixel decoder and world/action policy not trained; not completeRAE'],'protocol_sha256':g.interface.sha(O/'protocol.json')});print('COMPLETE',json.dumps(matrix),flush=True)
if __name__=='__main__':
 if '--check' in sys.argv:
  result=check_solver();write('solver-check.json',result);print(json.dumps(result))
 else:main()
