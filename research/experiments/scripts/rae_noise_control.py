"""Equal-noise training control, frozen encoders, existing evaluation vectors."""
import datetime,gc,hashlib,json,os,sys,time,zipfile
from pathlib import Path
import numpy as np,torch
from PIL import Image
import rae_interface_preflight as interface
sys.path.insert(0,str(Path(__file__).parent/'layers'))
import common as c
from readout_v2 import fit,predict,visual
R=c.ROOT;L=R/'results/layers-20260922';B=R/'results/rae-baseline-20260922';O=R/'results/rae-noise-control-20260922'
TASKS=c.TASKS;WN=['Wan48_native_scale','Wan48_per_token_layernorm'];DN=['L12_raw','L12_pca','L12_svae42','L12_svae43','L12_svae44'];NAMES=WN+DN;CONDS=['clean','noise0.10','noise0.04']
def write(name,obj):
 p=O/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def checks():
 rng=np.random.default_rng(712);x=rng.normal(size=(35,50));y=rng.normal(size=(35,2));xx=rng.normal(size=(10,50));a=fit(x,y,.01);b=fit(np.repeat(x,7,0),np.repeat(y,7,0),.07);delta=float(abs(predict(a,xx)-predict(b,xx)).max());assert delta<1e-10
 return {'passed':True,'repeated_clean_per_scene_regularization_prediction_max_abs':delta,'no_actual_model_refit':True}
def select_aug(x,y,ep):
 folds=np.array_split(np.random.default_rng(20260922).permutation(np.unique(ep)),5);scores=[]
 assert all((ep==e).sum()==7 for e in np.unique(ep))
 for alpha in [1e-4,.01,1.,100.]:
  losses=[]
  for fold in folds:
   va=np.isin(ep,fold);ratio=(~va).sum()/len(np.unique(ep[~va]));assert ratio==7
   m=fit(x[~va],y[~va],alpha*ratio);err=((predict(m,x[va])-y[va])/m['ys'])**2;losses.append(float(np.mean([err[ep[va]==e].mean() for e in fold])))
  scores.append({'alpha':alpha,'fold_scores':losses,'score':float(np.mean(losses))})
 choice=min(scores,key=lambda a:a['score']);return fit(x,y,choice['alpha']*7),{'selected_base_alpha':choice['alpha'],'effective_alpha':choice['alpha']*7,'all_scores':scores,'folds':[x.tolist() for x in folds],'variants_per_independent_episode':7}
@torch.inference_mode()
def wan_vectors(enc,im):
 z=enc.batch_encode(enc.preprocess_video([im])).float();assert tuple(z.shape)==(1,48,1,24,20)
 ln=torch.nn.functional.layer_norm(z.permute(0,2,3,4,1),(48,),eps=1e-6).permute(0,4,1,2,3)
 return [torch.nn.functional.adaptive_avg_pool2d(x[:,:,0],(2,2)).flatten(1)[0].cpu().numpy() for x in [z,ln]]
def main():
 assert not (O/'result.json').exists();start=time.monotonic();freeze=json.loads((O/'freeze.json').read_text())
 for path,h in freeze['files'].items():assert interface.sha(path)==h,path
 assert json.loads((O/'cpu-check.json').read_text())['passed'];torch.set_num_threads(4);torch.manual_seed(20260922);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 rows=json.loads((O/'clean-rows.json').read_text());br=json.loads((B/'train-rows.json').read_text());bi=[i for i,x in enumerate(br) if x['frame']==0];assert [br[i] for i in bi]==rows and len(rows)==105
 clean={k:v for k,v in np.load(O/'clean-dino-goal-vectors.npz').items()};v=np.load(B/'wan-training-vectors.npz');clean.update({k:v[k][bi] for k in WN});task=np.array([x['task'] for x in rows]);episode=np.array([x['episode'] for x in rows]);parity_indices=[int(np.flatnonzero(task==t)[0]) for t in TASKS]
 images=[]
 import cv2
 with zipfile.ZipFile(B/'train-images.zip') as z:
  for i,row in enumerate(rows):
   raw=z.read(str(bi[i])+'.jpg');assert hashlib.sha256(raw).hexdigest()==row['encoded_head_hashes'][0]
   images.append(c.resize_head(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)))
 weights=torch.load(R/'assets/wan22-vae-baseline/Wan2.2_VAE.pth',map_location='cpu',weights_only=True,mmap=True);weights=weights.get('model_state',weights);model=interface.WanVideoVAE38();model.load_state_dict({'model.'+k:v for k,v in weights.items()},strict=True,assign=True);wan=interface.WanVideoVAEEncoder(model.to('cuda',dtype=torch.bfloat16)).eval().requires_grad_(False)
 dino=c.encoder();parity_features=[];parity_vectors={k:[] for k in WN};torch.cuda.reset_peak_memory_stats()
 for i in parity_indices:
  for k,x in zip(WN,wan_vectors(wan,images[i])):parity_vectors[k].append(x)
  z=c.encode(dino,dino.preprocess_video([images[i]]*9));parity_features.append(z['L12'][0,:,:1].cpu().half())
 parity_vectors.update(visual(torch.stack(parity_features),'L12'));parity={}
 for k in NAMES:
  a=np.array(parity_vectors[k]);reference=clean[k][parity_indices];std=np.array([clean[k][task==rows[i]['task']].astype(float).std(0).clip(1e-6) for i in parity_indices]);delta=(a-reference)/std
  parity[k]={'max_abs':float(abs(a-reference).max()),'per_task_standardized_RMS':np.sqrt((delta**2).mean(1)).tolist(),'per_task_standardized_max':abs(delta).max(1).tolist()}
 write('hardware-parity.json',{'routes':parity,'training_only':True,'indices':parity_indices,'passed':all(max(v['per_task_standardized_RMS'])<=.05 and max(v['per_task_standardized_max'])<=.5 for v in parity.values())})
 assert json.loads((O/'hardware-parity.json').read_text())['passed'],'hardware/codec parity failed; preserve, no tolerance relaxation'
 del parity_features,parity_vectors;gc.collect();torch.cuda.empty_cache()
 noisyraw=[];wn={k:[] for k in WN};reference=[];condition=[]
 for i,row in enumerate(rows):
  arr=np.array(images[i])
  for sigma in [.1,.04]:
   for draw in range(3):
    rng=np.random.default_rng(20260922+row['episode_uid']*100000+row['frame']*10+draw);im=Image.fromarray(np.clip(arr.astype(np.float32)+rng.normal(0,255*sigma,arr.shape),0,255).astype(np.uint8));reference.append(i);condition.append(f'noise{sigma:.2f}')
    for k,x in zip(WN,wan_vectors(wan,im)):wn[k].append(x)
    z=c.encode(dino,dino.preprocess_video([im]*9));noisyraw.append(z['L12'][0,:,:1].cpu().half())
  if i%15==0:write('progress.json',{'stage':'training_noise_features','episodes':i+1,'elapsed_seconds':time.monotonic()-start});print('NOISY_TRAIN',i+1,flush=True)
 del wan,model,dino;gc.collect();torch.cuda.empty_cache();noisy=visual(torch.stack(noisyraw),'L12');noisy.update({k:np.array(v) for k,v in wn.items()});peak=torch.cuda.max_memory_allocated()/1e9;del noisyraw;gc.collect();torch.cuda.empty_cache()
 ref=np.concatenate([np.arange(105),np.array(reference)]);cond=np.array(['clean']*105+condition);arrays={k:np.concatenate([clean[k],noisy[k]]) for k in NAMES};assert len(ref)==735
 np.savez(O/'training-vectors.npz',**arrays,reference=ref,condition=cond);write('feature-complete.json',{'rows':735,'new_noisy_training_images':630,'cache_clean_images_reused':105,'peak_allocated_GB':peak,'hardware_parity_passed':True})
 goal=json.loads((L/'initial-goal/training-samples.json').read_text());lookup={(x['task'],x['episode']):x['target_xy'] for x in goal};y=np.array([lookup[x['task'],x['episode']] for x in rows],dtype=np.float32).astype(float);coeff={};selection={};models={}
 for name in NAMES:
  for t in TASKS:
   ii=task[ref]==t;key=name+'/'+t;m,choice=select_aug(arrays[name][ii].astype(float),y[ref[ii]],episode[ref[ii]]);models[key]=m;selection[key]=choice
   for k,v in m.items():coeff[key+'/'+k]=v
   print('FIT',key,choice['selected_base_alpha'],flush=True)
 np.savez(O/'coefficients.npz',**coeff);write('selection.json',selection);write('fit-complete.json',{'completed_at':datetime.datetime.now().astimezone().isoformat(),'fits':21,'eval_vectors_or_labels_accessed':False})
 # All models are now fixed. Reuse old evaluation vectors; no encoder execution.
 evd=np.load(L/'confirmation/vectors.npz');evw=np.load(B/'wan-evaluation-vectors.npz');erows=json.loads((L/'confirmation/sample-manifest.json').read_text());etask=np.array([x['task'] for x in erows]);eep=np.array([x['seed'] for x in erows]);eframe=np.array([x['frame'] for x in erows]);gl={(x['task'],x['seed']):x['target_xy'] for x in json.loads((L/'initial-goal/fresh-labels.json').read_text())};constants=json.loads((L/'initial-goal/constants.json').read_text());results={};preds={}
 for name in NAMES:
  data=evw if name in WN else evd;rr=data['reference'];cc=data['condition']
  for t in TASKS:
   m=models[name+'/'+t]
   for con in CONDS:
    ii=np.flatnonzero((etask[rr]==t)&(eframe[rr]==0)&(cc==con));ri=rr[ii];pred=predict(m,data[name][ii].astype(float));target=np.array([gl[t,int(eep[x])] for x in ri]);scale=np.array(constants[t]['std']);error=((pred-target)/scale)**2;eps=np.unique(eep[ri]);assert len(eps)==20;per=np.array([error[eep[ri]==e].mean(0) for e in eps]);key=name+'/'+t+'/'+con;results[key]={'episode_ids':eps.tolist(),'per_episode_xy_nmse':per.tolist(),'E':float(per.mean())};preds[key]=pred
 groups={WN[0]:[WN[0]],WN[1]:[WN[1]],'DINO_raw768':['L12_raw'],'DINO_PCA48':['L12_pca'],'DINO_SVAE48_seed_mean':['L12_svae42','L12_svae43','L12_svae44']};oldw=json.loads((B/'probe-result.json').read_text())['conditions'];oldd=json.loads((L/'initial-goal/fresh-summary.json').read_text())['conditions'];bootstrap=np.random.default_rng(20260922).integers(0,20,(2000,3,20));matrix={};within={};between={}
 def compare(a,b):
  diff=a-b;ds=np.stack([diff[t,bootstrap[:,t]] for t in range(3)],1).mean(axis=(1,2,3));bs=np.stack([b[t,bootstrap[:,t]] for t in range(3)],1).mean(axis=(1,2,3));return {'relative_change_percent':float(100*diff.mean()/b.mean()),'descriptive95':np.quantile(100*ds/bs,[.025,.975]).tolist(),'per_task_relative_percent':(100*diff.mean((1,2))/b.mean((1,2))).tolist()}
 for con in CONDS:
  a={};b={}
  for label,names in groups.items():
   aa=[];bb=[]
   for t in TASKS:
    new=[results[n+'/'+t+'/'+con] for n in names];old=[oldw[f'goal/{n}/{t}/{con}'] if n in WN else oldd[f'ridge/{t}/{n}/{con}'] for n in names];assert all(x['episode_ids']==new[0]['episode_ids'] for x in new+old);aa.append(np.mean([x['per_episode_xy_nmse'] for x in new],0));bb.append(np.mean([x['per_episode_xy_nmse'] for x in old],0))
   a[label]=np.array(aa);b[label]=np.array(bb)
  matrix[con]={n:{'noise_adapted_E':float(a[n].mean()),'original_clean_trained_E':float(b[n].mean()),'noise_adapted_per_task_E':a[n].mean((1,2)).tolist()} for n in groups};within[con]={n:compare(a[n],b[n]) for n in groups};between[con]={base:{n:compare(a[n],a[base]) for n in ['DINO_raw768','DINO_PCA48','DINO_SVAE48_seed_mean']} for base in WN}
 np.savez(O/'predictions.npz',**preds);write('result.json',{'completed_at':datetime.datetime.now().astimezone().isoformat(),'scope':'Exploratory noise-adapted goal readouts on exposed60; full RAE and closed-loop control not evaluated','training_episodes':105,'evaluation_episodes':60,'new_readout_fits':21,'task_order':TASKS,'conditions':results,'matrix':matrix,'within_route_noise_adapted_minus_original':within,'between_noise_adapted_DINO_minus_Wan':between,'elapsed_seconds':time.monotonic()-start,'peak_allocated_GB':peak,'noise_training_evaluation_same_severities':True,'limits':['goal only: no short-state or clean-state noninferiority conclusion','noise strengths and old evaluation outcomes were previously exposed','all routes use identical training-only noise and per-scene regularization','raw dimension differs from192D compressed readout controls','encoder/objective/architecture differences do not isolate pretraining causally','no pixel decoder trained for DINO; frozen feature path is RAE-style only'],'protocol_sha256':interface.sha(O/'protocol.json')});print('COMPLETE',json.dumps(matrix),flush=True)
if __name__=='__main__':
 if '--check' in sys.argv:print(json.dumps(checks()))
 else:main()
