"""Fresh native multiview compression fits on the105 training episodes only."""
import datetime, gc, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from openwam.dataloader.registry import build_dataset
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
from openwam.model.video_backbone.encoder.svae.model import SVAE,svae_loss
R=Path('/data02/zifanz4/openwam-experiments');E=R/'endtoend-20260923';O=E/'compressors-multiview'
def write(name,d):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def guard():
 for p in [Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json'),Path('/home/zifanz4/openwam-experiments/studies/endtoend-20260923/paused.json')]:assert not p.exists(),p
 assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(plan['deadline'])
plan=json.loads((O/'protocol.json').read_text());assert not (O/'complete.json').exists();guard()
torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
cfg=OmegaConf.load(E/'templates/svae/config.yaml');cfg.dataloader.multiview=True
assert cfg.dataloader.color_jitter.enabled is False
dataset=build_dataset(cfg.dataloader)
enc=DinoV3VideoEncoder.from_skeleton({},ckpt_dir=str(E/'templates/pca'))
with safe_open(str(R/'assets/dinov3-study/encoder.safetensors'),framework='pt',device='cpu') as f:
 prefix='video_backbone.video_encoder._m.';weights={n[len(prefix):]:f.get_tensor(n) for n in f.keys() if n.startswith(prefix)}
 enc._m.load_state_dict(weights,strict=True)
del weights
enc.cuda().eval().requires_grad_(False)
manifest=[];parts=[];groups=[];begin=time.monotonic()
assert len(dataset._sub_datasets)==3
for ti,ds in enumerate(dataset._sub_datasets):
 assert len(ds._episode_files)==35
 group=[]
 for ei,path in enumerate(ds._episode_files):
  guard();n=ds._episode_lengths[ei];starts=np.unique(np.linspace(0,n-33,12,dtype=int));assert len(starts)==12
  features=[];rows=[]
  for st in starts:
   sample=ds._build_sample(ei,int(st));assert len(sample['video'])==9
   with torch.inference_mode():z=enc.batch_encode_pooled_for_svae_training(enc.preprocess_video(sample['video']))
   assert z.shape==(1,768,3,24,20) and torch.isfinite(z).all(),z.shape
   features.append(z[0].cpu().half())
   rows.append({'task':ds.task_name,'path':str(path),'start':int(st),'frame_indices':list(range(int(st),int(st)+33,4)),'multiview_composite_sha256':[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in sample['video']]})
  dest=O/'features'/str(ti)/f'{Path(path).stem}.pt';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
  torch.save({'features':torch.stack(features),'manifest':rows},dest)
  group.extend(range(len(manifest),len(manifest)+12));manifest.extend(rows);parts.append(torch.stack(features))
  write('extraction-progress.json',{'time':now(),'episodes':len(manifest)//12,'clips':len(manifest),'seconds':time.monotonic()-begin})
  print('EXTRACT',ti,ei,len(manifest),flush=True)
 groups.append(torch.tensor(group))
assert len(manifest)==1260;write('feature-manifest.json',manifest);write('extraction-complete.json',{'time':now(),'episodes':105,'clips':1260,'native_multiview':True,'no_validation_or_test_frames':True,'seconds':time.monotonic()-begin})
del enc,dataset;gc.collect();torch.cuda.empty_cache();x=torch.cat(parts);del parts
n=0;su=torch.zeros(768,device='cuda',dtype=torch.float64);ss=su.clone();indices=torch.arange(len(x))
for ids in indices.split(3):
 z=x[ids].cuda().movedim(1,-1).reshape(-1,768).double();n+=len(z);su+=z.sum(0);ss+=(z*z).sum(0)
mu=(su/n).float();std=(ss/n-(su/n)**2).clamp_min(1e-12).sqrt().float();cov=torch.zeros(768,768,device='cuda')
for ids in indices.split(3):
 z=(x[ids].cuda().movedim(1,-1).reshape(-1,768).float()-mu)/std;cov+=z.T@z
values,vectors=torch.linalg.eigh((cov/n).double().cpu());pca={'mean':mu.cpu(),'std':std.cpu(),'basis':vectors.flip(1).float()[:,:48],'eigenvalues':values.flip(0).float(),'train_tokens':n}
torch.save(pca,O/'pca.pt');assert (pca['eigenvalues'][:48]>0).all()
write('pca-complete.json',{'time':now(),'train_tokens':n,'sha256':sha(O/'pca.pt'),'retained_standardized_variance':float(values[-48:].sum()/values.sum())})
config=json.loads((R/'assets/dinov3-study/svae_config.json').read_text())['model_config']
torch.manual_seed(42);m=SVAE(**config).cuda();m.set_input_stats(pca['mean'],pca['std'])
opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4);rng=torch.Generator().manual_seed(42);begin=time.monotonic();steps=plan['svae_steps']
for step in range(1,steps+1):
 guard();selected=torch.stack([idx[torch.randint(len(idx),(1,),generator=rng)[0]] for idx in groups]);m.train();opt.zero_grad(set_to_none=True)
 lr=1e-4*min(1.,step/20)*(.95+.05*np.cos(np.pi*step/steps))
 for g in opt.param_groups:g['lr']=lr
 with torch.autocast('cuda',dtype=torch.bfloat16):output=m(x[selected].cuda().float());loss,stats=svae_loss(output,beta=1e-4*min(1.,step/400),cos_weight=1.)
 assert torch.isfinite(loss),step
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step()
 if step%20==0:
  row={'time':now(),'step':step,'loss':float(loss),'gradient_norm':float(norm),'seconds':time.monotonic()-begin}
  with (O/'svae-training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print('FIT',row,flush=True)
guard();m.eval();torch.save({'format_version':2,'model_config':m.config_dict(),'state_dict':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'step':steps,'seed':42,'source':'native_multiview_105train'},O/'svae-final.pt')
write('complete.json',{'time':now(),'episodes':105,'clips':1260,'svae_steps':steps,'svae_seed':42,'selection':'fixed final step; no checkpoint or representation selection','pca_sha256':sha(O/'pca.pt'),'svae_sha256':sha(O/'svae-final.pt'),'feature_manifest_sha256':sha(O/'feature-manifest.json'),'policy_seed_effect_separate_from_fixed_reducer_seed':True})
