"""Fixed train-only temporal kernel fit; reconstruction heads never enter policy."""
import datetime,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from safetensors import safe_open
from openwam.dataloader.registry import build_dataset
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder,_causal_temporal_pool
from openwam.model.video_backbone.encoder.dinov3_temporal import ConvexTemporalPool
from openwam.model.video_backbone.encoder.svae.model import load_svae
from check_pool import check
R=Path('/data02/zifanz4/openwam-experiments');E=R/'endtoend-20260923';N=R/'temporal-20260926';O=N/'fit';P=Path('/home/zifanz4/openwam-experiments');S=P/'studies/temporal-20260926'
plan=json.loads((S/'protocol.json').read_text())
def now():return datetime.datetime.now().astimezone().isoformat()
def write(name,obj):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def guard():
 assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(plan['deadline'])
 for p in [S/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),p

def norm(z):return torch.nn.functional.layer_norm(z.movedim(1,-1),(48,),eps=1e-6).movedim(-1,1)
class ResidualHead(nn.Module):
 def __init__(self):
  super().__init__();self.proj=nn.Linear(48,4*768);nn.init.zeros_(self.proj.weight);nn.init.zeros_(self.proj.bias)
 def forward(self,z):
  b,c,t,h,w=z.shape
  residual=self.proj(z.movedim(1,-1)).reshape(b,t,h,w,4,768).permute(0,5,1,4,2,3)
  return residual-residual.mean(3,keepdim=True)

def main():
 guard();assert not (O/'started.json').exists();O.mkdir(exist_ok=True);write('started.json',{'time':now(),'protocol':plan})
 torch.set_num_threads(4);torch.manual_seed(426);torch.backends.cuda.matmul.allow_tf32=False
 write('pool-contracts.json',check())
 cfg=OmegaConf.load(E/'templates/svae/config.yaml');cfg.dataloader.multiview=True;assert cfg.dataloader.color_jitter.enabled is False
 ds=build_dataset(cfg.dataloader);enc=DinoV3VideoEncoder.from_skeleton({},ckpt_dir=str(E/'templates/pca'))
 with safe_open(str(R/'assets/dinov3-study/encoder.safetensors'),framework='pt',device='cpu') as f:
  pre='video_backbone.video_encoder._m.';weights={n[len(pre):]:f.get_tensor(n) for n in f.keys() if n.startswith(pre)};enc._m.load_state_dict(weights,strict=True)
 del weights;enc.cuda().eval().requires_grad_(False);pool=ConvexTemporalPool(768).cuda();parts=[];groups=[];manifest=[];begin=time.monotonic()
 for ti,sub in enumerate(ds._sub_datasets):
  assert len(sub._episode_files)==35;ids=[]
  for ei,path in enumerate(sub._episode_files):
   guard();starts=np.unique(np.linspace(0,sub._episode_lengths[ei]-33,6,dtype=int));assert len(starts)==6
   feat=[];rows=[]
   for st in starts:
    sample=sub._build_sample(ei,int(st));assert len(sample['video'])==9
    with torch.inference_mode():
     pixels=enc.preprocess_video(sample['video']);grid=enc._batch_encode_frame_grid(pixels)
     assert grid.shape==(1,768,9,24,20) and torch.isfinite(grid).all()
     if not manifest and not rows:
      original=enc._batch_encode_pooled_raw(pixels);candidate=pool(grid)
      assert torch.equal(original,candidate)
      write('native-mean-parity.json',{'time':now(),'passed':True,'max_abs':float((original-candidate).abs().max()),'dtype':str(grid.dtype),'shape':list(grid.shape),'data':'first training clip'})
    feat.append(grid[0].cpu().half());rows.append({'task':sub.task_name,'path':str(path),'start':int(st),'frame_indices':list(range(int(st),int(st)+33,4)),'rgb_sha256':[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in sample['video']]})
   data=torch.stack(feat);dest=O/'features'/str(ti)/f'{Path(path).stem}.pt';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();torch.save({'features':data,'manifest':rows},dest)
   ids.extend(range(len(manifest),len(manifest)+6));manifest.extend(rows);parts.append(data)
   write('progress.json',{'time':now(),'phase':'extraction','episodes':len(manifest)//6,'clips':len(manifest),'seconds':time.monotonic()-begin});print('EXTRACT',len(manifest)//6,len(manifest),flush=True)
  groups.append(torch.tensor(ids))
 assert len(manifest)==630;write('feature-manifest.json',manifest)
 del enc,ds,grid,pixels;gc.collect();torch.cuda.empty_cache();x=torch.cat(parts);del parts;gc.collect()
 svae=load_svae(str(E/'compressors-multiview/svae-final.pt')).cuda().float().eval().requires_grad_(False)
 heads=[ResidualHead().cuda(),ResidualHead().cuda()];opts=[torch.optim.AdamW(list(pool.parameters())+list(heads[0].parameters()),lr=.001,betas=(.9,.99),weight_decay=0),torch.optim.AdamW(heads[1].parameters(),lr=.001,betas=(.9,.99),weight_decay=0)]
 std=svae.input_std[None,:,None,None,None];mean=svae.input_mean[None,:,None,None,None]
 rng=torch.Generator().manual_seed(426);start=time.monotonic()
 for step in range(1,1501):
  guard();ids=torch.stack([g[torch.randint(len(g),(1,),generator=rng)[0]] for g in groups]);batch=x[ids].cuda().float();target=((batch[:,:,1:]-mean)/std).reshape(3,768,2,4,24,20)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reference=norm(svae.encode_mean(_causal_temporal_pool(batch))).float()
  stats=[]
  for arm in [0,1]:
   opts[arm].zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    pooled=pool(batch) if arm==0 else _causal_temporal_pool(batch)
    z=svae.encode_mean(pooled);zn=norm(z);base=(svae.decode(z).float()-mean)/std
    prediction=base[:,:,1:,None]+heads[arm](zn[:,:,1:]).float()
    reconstruction=(prediction-target).square().mean();motion=((prediction[:,:,:,1:]-prediction[:,:,:,:-1])-(target[:,:,:,1:]-target[:,:,:,:-1])).square().mean()
    anchor=(zn.float()-reference).square().mean();w=pool.weights();kl=(w*(w*4).log()).sum(-1).mean() if arm==0 else z.new_tensor(0.)
    loss=reconstruction+motion+.05*anchor+.01*kl
   assert torch.isfinite(loss);loss.backward()
   if arm==0:
    grad=pool.logits.grad;assert grad is not None and grad.isfinite().all()
    if step==1:
     assert grad.abs().sum()>0;write('gradient-gate.json',{'passed':True,'kernel_gradient_norm':float(grad.norm()),'reducer_parameters_frozen':not any(p.requires_grad for p in svae.parameters())})
   params=list(pool.parameters())+list(heads[0].parameters()) if arm==0 else list(heads[1].parameters());gn=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(gn);opts[arm].step()
   stats.append({'loss':float(loss),'reconstruction':float(reconstruction),'motion':float(motion),'anchor':float(anchor),'kl':float(kl),'gradient_norm':float(gn)})
  if step%20==0 or step==1:
   row={'time':now(),'step':step,'learned':stats[0],'mean_shadow':stats[1],'seconds':time.monotonic()-start,'weight_min':float(pool.weights().min()),'weight_max':float(pool.weights().max())}
   with (O/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   write('progress.json',dict(row,phase='fit'));print('FIT',json.dumps(row),flush=True)
 guard();saved={'format_version':1,'pool_state_dict':{k:v.detach().cpu() for k,v in pool.state_dict().items()},'step':1500,'seed':426};torch.save(saved,O/'temporal-final.pt')
 torch.save({'learned_head':heads[0].state_dict(),'mean_shadow_head':heads[1].state_dict()},O/'diagnostic-heads-final.pt')
 write('complete.json',{'time':now(),'steps':1500,'episodes':105,'clips':630,'fixed_final':True,'pool_sha256':hashlib.sha256((O/'temporal-final.pt').read_bytes()).hexdigest(),'pool_parameters':sum(p.numel() for p in pool.parameters()),'last_training_metrics':stats,'policy_success_measured':False,'heads_removed_from_policy':True})
if __name__=='__main__':main()
