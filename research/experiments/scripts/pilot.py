#!/usr/bin/env python3
"""Bounded, episode-disjoint OpenWAM encoder/compression pilot (not policy evaluation)."""
import argparse, importlib, json, os, sys, time, types, hashlib, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import h5py, cv2
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'OpenWAM';ASSET=ROOT/'assets/dinov3-study';OUT=ROOT/'results/pilot-adjust-bottle-v1'
OUT.mkdir(parents=True,exist_ok=True)
# Import the unmodified upstream leaf modules, without eager full-WAM registration.
for package in ['openwam','openwam.model','openwam.model.video_backbone','openwam.model.video_backbone.encoder','openwam.dataloader','openwam.dataloader.transforms']:
 m=types.ModuleType(package);m.__path__=[str(REPO.joinpath(*package.split('.')))];sys.modules[package]=m
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
from openwam.model.video_backbone.encoder.svae.model import SVAE,svae_loss
from openwam.dataloader.transforms.multiview import assemble_multiview_layout

SEED=42;TRAIN_SEED=42;NCLIPS=12
DATA_TASK='adjust_bottle'
CAMS=['head_camera','left_camera','right_camera']
torch.set_num_threads(6);torch.manual_seed(SEED);np.random.seed(SEED)
torch.backends.cuda.matmul.allow_tf32=False

def write_json(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def ensure_gpu():
 assert torch.cuda.is_available(),'CUDA unavailable'
 print('GPU',torch.cuda.get_device_name(0),'torch',torch.__version__,flush=True)
def build_encoder():
 cfg=AutoConfig.from_pretrained(ASSET/'dinov3')
 model=AutoModel.from_config(cfg,dtype=torch.bfloat16)
 weights=load_file(ASSET/'encoder.safetensors')
 prefix='video_backbone.video_encoder._m.'
 state={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
 model.load_state_dict(state,strict=True)
 enc=DinoV3VideoEncoder(model,embed_dim=cfg.hidden_size,patch_size=cfg.patch_size,num_register_tokens=cfg.num_register_tokens).cuda().eval()
 enc.requires_grad_(False)
 return enc

def extract():
 if (OUT/'features.pt').exists():
  print('Reuse completed feature cache',flush=True);return
 ensure_gpu();enc=build_encoder()
 files=sorted((ROOT/'data'/DATA_TASK).rglob('*.hdf5'),key=lambda p:int(p.stem.replace('episode','')))
 assert len(files)==50
 rng=np.random.default_rng(SEED);order=rng.permutation(50)
 splits={int(i):'train' if pos<35 else 'val' if pos<40 else 'test' for pos,i in enumerate(order)}
 manifest=[];features=[];states=[];targets=[];timings=[]
 start=time.perf_counter();checks={};torch.cuda.reset_peak_memory_stats()
 for ei,path in enumerate(files):
  with h5py.File(path,'r') as f:
   n=len(f['endpose/left_endpose']);assert n>=33
   starts=np.unique(np.linspace(0,n-33,NCLIPS,dtype=int))
   for st in starts:
    frames=[]
    for t in range(st,st+33,4):
     images={cam:Image.fromarray(cv2.imdecode(np.frombuffer(bytes(f[f'observation/{cam}/rgb'][t]),np.uint8),cv2.IMREAD_COLOR)) for cam in CAMS}
     frames.append(assemble_multiview_layout(images,CAMS,384,320))
    if not manifest:frames[0].save(OUT/'input-preview.png')
    pix=enc.preprocess_video(frames)
    torch.cuda.synchronize();t0=time.perf_counter()
    with torch.inference_mode():z=enc.batch_encode_pooled_for_svae_training(pix)
    torch.cuda.synchronize();timings.append(time.perf_counter()-t0)
    assert tuple(z.shape)==(1,768,3,24,20)
    assert torch.isfinite(z).all()
    if not manifest:
     with torch.inference_mode():
      altered=pix.clone();altered[:,:,1:]=altered[:,:,1:].flip(2)
      zz=enc.batch_encode_pooled_for_svae_training(altered)
     checks['condition_future_invariance_max_abs']=float((z[:,:,0]-zz[:,:,0]).abs().max())
     assert checks['condition_future_invariance_max_abs']==0.0
     checks['strict_encoder_weights_loaded']=True
     checks['upstream_encoder_class']=str(type(enc))
    features.append(z[0].cpu().half())
    lp=f['endpose/left_endpose'][st];rp=f['endpose/right_endpose'][st]
    lg=float(f['endpose/left_gripper'][st]);rg=float(f['endpose/right_gripper'][st])
    future=st+4
    # Target is future achieved-state change, not a controller command.
    target=np.concatenate([f['endpose/left_endpose'][future,:3]-lp[:3],f['endpose/right_endpose'][future,:3]-rp[:3],[f['endpose/left_gripper'][future],f['endpose/right_gripper'][future]]])
    states.append(np.concatenate([lp,rp,[lg,rg]]));targets.append(target)
    manifest.append({'episode':ei,'episode_file':str(path.relative_to(ROOT/'data')),'start_frame':int(st),'frame_indices':list(range(int(st),int(st)+33,4)),'split':splits[ei]})
  print('EXTRACT',ei+1,'/50 clips',len(manifest),'elapsed_s',round(time.perf_counter()-start,1),flush=True)
 data={'features':torch.stack(features),'state':torch.tensor(np.array(states),dtype=torch.float32),'target':torch.tensor(np.array(targets),dtype=torch.float32),'manifest':manifest}
 torch.save(data,OUT/'features.tmp');(OUT/'features.tmp').replace(OUT/'features.pt')
 write_json(OUT/'manifest.json',manifest)
 for s in ['train','val','test']:
  checks[s+'_episodes']=sorted(set(m['episode']for m in manifest if m['split']==s))
 assert not(set(checks['train_episodes'])&set(checks['test_episodes']))
 checks.update({'clips':len(manifest),'feature_shape':list(data['features'].shape),'extract_wall_seconds':time.perf_counter()-start,'median_encoder_seconds_per_clip':float(np.median(timings[5:])),'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'peak_reserved_gb':torch.cuda.max_memory_reserved()/1e9,'feature_cache_bytes':(OUT/'features.pt').stat().st_size,'seed':SEED,'clips_per_episode':NCLIPS})
 write_json(OUT/'extraction.json',checks)

def get_data():return torch.load(OUT/'features.pt',map_location='cpu',weights_only=False)
def ids(data,split):return torch.tensor([i for i,m in enumerate(data['manifest']) if m['split']==split])
def flat(x):return x.permute(0,2,3,4,1).reshape(-1,x.shape[1]).float()
def fit_pca(data):
 p=OUT/'pca.pt'
 if p.exists():return torch.load(p,weights_only=True)
 x=data['features'];ii=ids(data,'train');s=torch.zeros(768,device='cuda',dtype=torch.float64);ss=s.clone();n=0
 for part in ii.split(4):
  z=flat(x[part].cuda()).double();s+=z.sum(0);ss+=(z*z).sum(0);n+=z.shape[0]
 mean=(s/n).float();std=(ss/n-(s/n)**2).clamp_min(1e-12).sqrt().float()
 cov=torch.zeros(768,768,device='cuda')
 for part in ii.split(4):
  z=(flat(x[part].cuda())-mean)/std;cov+=z.T@z
 cov/=n
 values,basis=torch.linalg.eigh(cov.double().cpu());values=values.flip(0).float();basis=basis.flip(1).float()
 result={'mean':mean.cpu(),'std':std.cpu(),'eigenvalues':values,'basis':basis,'train_tokens':n,'train_episode_ids':sorted(set(data['manifest'][i]['episode']for i in ii.tolist()))}
 torch.save(result,p);print('PCA fit',n,'training tokens',flush=True);return result

def make_svae(pca,published=False):
 cfg=json.loads((ASSET/'svae_config.json').read_text())['model_config'];m=SVAE(**cfg)
 if published:
  sd=load_file(ASSET/'encoder.safetensors');prefix='video_backbone.video_encoder._svae.'
  m.load_state_dict({k[len(prefix):]:v.float()for k,v in sd.items()if k.startswith(prefix)},strict=True)
 else:m.set_input_stats(pca['mean'],pca['std'])
 return m.cuda()

@torch.no_grad()
def validation_mse(model,x,ii):
 model.eval();numerator=0.;count=0
 for part in ii.split(2):
  z=x[part].cuda().float()
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(z)
  err=(out['recon'].float()-out['target'].float()).square()
  numerator+=float(err.sum());count+=err.numel()
 return numerator/count

def train_svae(data,pca,steps):
 path=OUT/f'svae-fresh-{steps}.pt'
 if path.exists():return path
 torch.manual_seed(TRAIN_SEED);m=make_svae(pca);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
 train_ids=ids(data,'train');val_ids=ids(data,'val');x=data['features'];rng=torch.Generator().manual_seed(TRAIN_SEED)
 hist=[];best=float('inf');start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 initial=validation_mse(m,x,val_ids);print('SVAE initial val_mse',initial,flush=True)
 for step in range(1,steps+1):
  selected=train_ids[torch.randint(len(train_ids),(2,),generator=rng)]
  z=x[selected].cuda().float();m.train();opt.zero_grad(set_to_none=True)
  lr=1e-4*min(1.0,step/20)*(0.95+0.05*np.cos(np.pi*step/steps))
  for group in opt.param_groups:group['lr']=lr
  with torch.autocast('cuda',dtype=torch.bfloat16):
   out=m(z);loss,stat=svae_loss(out,beta=1e-4*min(1.,step/(steps*.2)),cos_weight=1.)
  assert torch.isfinite(loss)
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step%50==0 or step==steps:
   val=validation_mse(m,x,val_ids)
   rec={'step':step,'train_loss':float(loss),'train_mse':float(stat['mse']),'val_mse':val,'elapsed_s':time.perf_counter()-start}
   hist.append(rec);print('SVAE',json.dumps(rec),flush=True)
   if val<best:
    best=val
    torch.save({'format_version':2,'model_config':m.config_dict(),'state_dict':{k:v.detach().cpu().clone()for k,v in m.state_dict().items()},'step':step,'val_mse':val},path)
   write_json(OUT/'training-curve.json',hist)
 write_json(OUT/'training.json',{'steps':steps,'best_val_mse':best,'initial_val_mse':initial,'wall_seconds':time.perf_counter()-start,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'seed':TRAIN_SEED,'batch_clips':2,'learning_rate':1e-4,'note':'short-budget pilot, not convergence-matched to official S-VAE'})
 return path

def representation(z,name,pca,model=None):
 if name=='raw':return z,None
 if name.startswith('pca'):
  d=int(name[3:]);mean=pca['mean'].cuda();std=pca['std'].cuda();basis=pca['basis'][:,:d].cuda()
  b,c,t,h,w=z.shape;xx=(flat(z)-mean)/std;lat=xx@basis;rec=(lat@basis.T)*std+mean
  return lat.reshape(b,t,h,w,d).permute(0,4,1,2,3),rec.reshape(b,t,h,w,c).permute(0,4,1,2,3)
 with torch.autocast('cuda',dtype=torch.bfloat16):lat=model.encode_mean(z);rec=model.decode(lat)
 return lat.float(),rec.float()

def probe_metrics(pred,target,scale):
 err=pred-target
 return {'normalized_mse':float(np.mean((err/scale)**2)),'translation_rmse_mm':float(1000*np.sqrt(np.mean(err[:,:6]**2))),'gripper_rmse':float(np.sqrt(np.mean(err[:,6:]**2)))}

def ridge_probe(vectors,data,tag):
 train=ids(data,'train').numpy();val=ids(data,'val').numpy();test=ids(data,'test').numpy()
 target=data['target'].numpy().astype(np.float64);ym=target[train].mean(0);ys=target[train].std(0).clip(1e-6)
 xx=np.asarray(vectors,dtype=np.float64);xm=xx[train].mean(0);xs=xx[train].std(0).clip(1e-5);xx=(xx-xm)/xs
 xx/=np.sqrt(xx.shape[1]);yt=(target[train]-ym)/ys
 kernel=xx[train]@xx[train].T;ev,q=np.linalg.eigh(kernel);ev=ev.clip(0);proj=q.T@yt
 best=None
 for alpha in [1e-5,1e-4,1e-3,1e-2,.1,1.,10.]:
  coef=q@(proj/(ev[:,None]+alpha));pv=(xx[val]@xx[train].T)@coef*ys+ym
  score=probe_metrics(pv,target[val],ys)['normalized_mse']
  if best is None or score<best[0]:best=(score,alpha,coef)
 pred=(xx[test]@xx[train].T)@best[2]*ys+ym
 res={'condition':tag,'alpha':best[1],'validation_normalized_mse':best[0],**probe_metrics(pred,target[test],ys)}
 # Cluster bootstrap over held-out episodes; windows are not independent samples.
 episodes=np.array([data['manifest'][i]['episode']for i in test]);unique=np.unique(episodes);rng=np.random.default_rng(SEED)
 per_ep=np.array([np.mean(((pred[episodes==e]-target[test][episodes==e])/ys)**2)for e in unique]);means=np.mean(rng.choice(per_ep,(1000,len(unique)),replace=True),axis=1)
 res['normalized_mse_episode_bootstrap_95ci']=[float(x)for x in np.quantile(means,[.025,.975])]
 return res

@torch.no_grad()
def evaluate(data,pca,path):
 checkpoint=torch.load(path,weights_only=True);fresh=make_svae(pca);fresh.load_state_dict(checkpoint['state_dict']);fresh.eval()
 published=make_svae(pca,published=True).eval();x=data['features'];test=ids(data,'test');testset=set(test.tolist());summaries=[];probes=[];state=data['state'].numpy()
 target=data['target'].numpy();tr=ids(data,'train').numpy();te=test.numpy();ys=target[tr].std(0).clip(1e-6)
 persistent=np.zeros_like(target[te]);persistent[:,6:]=state[te,-2:]
 baselines={'zero_motion_current_gripper':probe_metrics(persistent,target[te],ys),'training_mean':probe_metrics(np.tile(target[tr].mean(0),(len(te),1)),target[te],ys)}
 probes.append(ridge_probe(state,data,'proprio_only'))
 for name in ['raw','pca24','pca48','pca96','svae_fresh48','svae_published48']:
  model=fresh if name=='svae_fresh48'else published if name=='svae_published48'else None
  vectors=[];errs=[];cosines=[];torch.cuda.synchronize();t0=time.perf_counter()
  for start in range(0,len(x),2):
   z=x[start:start+2].cuda().float();lat,rec=representation(z,name,pca,model)
   # Exactly the current conditioning frame; never future target latents.
   cur=lat[:,:,0];cur=F.layer_norm(cur.permute(0,2,3,1),(cur.shape[1],),eps=1e-6).permute(0,3,1,2)
   vectors.append(F.adaptive_avg_pool2d(cur,(2,2)).flatten(1).cpu().numpy())
   if rec is not None:
    for j in range(len(z)):
     if start+j in testset:
      std=pca['std'].cuda()[:,None,None,None]
      errs.append(float(((rec[j]-z[j])/std).square().mean()))
      cosines.append(float(F.cosine_similarity(flat(rec[j:j+1]),flat(z[j:j+1]),dim=-1).mean()))
  torch.cuda.synchronize();seconds=time.perf_counter()-t0
  vec=np.concatenate(vectors);row={'condition':name,'channels':768 if name=='raw'else int(name[3:])if name.startswith('pca')else 48,'eval_seconds_all_clips':seconds,'test_standardized_feature_mse':float(np.mean(errs))if errs else 0.,'test_feature_cosine':float(np.mean(cosines))if cosines else 1.,'published_training_overlap_possible':name=='svae_published48'}
  if name.startswith('pca'):row['train_variance_retained']=float(pca['eigenvalues'][:row['channels']].sum()/pca['eigenvalues'].sum())
  summaries.append(row)
  probes.append(ridge_probe(vec,data,name+'_visual_only'))
  probes.append(ridge_probe(np.concatenate([vec,state],axis=1),data,name+'_visual_plus_proprio'))
  print('EVAL',json.dumps(row),json.dumps(probes[-1]),flush=True)
  write_json(OUT/'metrics.json',{'reconstruction':summaries,'probes':probes,'baselines':baselines,'selected_svae_step':checkpoint['step'],'limitations':['one task and one training seed','600 correlated windows across 50 episodes, episode-disjoint splits','short-budget S-VAE training, not converged comparison','published S-VAE may have trained on all of these trajectories; contextual reference only','state-change probe is not an action policy or closed-loop success metric'],'target':'4-raw-step achieved end-effector translation delta (6D) and future achieved gripper (2D)'})

def main():
 global OUT,TRAIN_SEED
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','extract','fit'],default='all');p.add_argument('--steps',type=int,default=500);p.add_argument('--train-seed',type=int,default=42);p.add_argument('--run-tag',default='');a=p.parse_args()
 TRAIN_SEED=a.train_seed
 if a.run_tag:
  base=OUT;OUT=base.parent/('pilot-adjust-bottle-'+a.run_tag);OUT.mkdir(exist_ok=True)
  for filename in ['features.pt','pca.pt','manifest.json','extraction.json']:
   link=OUT/filename
   if not link.exists():link.symlink_to(base/filename)
 start=time.perf_counter()
 if a.stage in ['all','extract']:extract()
 if a.stage in ['all','fit']:
  ensure_gpu();data=get_data();pca=fit_pca(data);path=train_svae(data,pca,a.steps);evaluate(data,pca,path)
 write_json(OUT/'last-run.json',{'stage':a.stage,'wall_seconds':time.perf_counter()-start,'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
