"""Experiment harness: native trainer, shared random trainables, frozen pretrained encoders.

No published action/world weights enter this controlled comparison. The loader
hook avoids writing nine redundant copies of the same frozen text encoder before
training. Saved outputs remain ordinary self-contained native checkpoints.
"""
import gc, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from omegaconf import OmegaConf
R=Path('/data02/zifanz4/openwam-experiments');E=R/'endtoend-20260923'
RUN=Path(os.environ['OPENWAM_RUN_DIR']);RUN.mkdir(parents=True,exist_ok=True)
from openwam.train.utils import ckpt_model_loader as loader
original_build=loader.build_architecture_from_ckpt_dir

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def record(name,data):(RUN/name).write_text(json.dumps(data,indent=2)+'\n')
def load_prefix(module,path,prefix):
 target=module.state_dict()
 with safe_open(str(path),framework='pt',device='cpu') as f:
  names={n[len(prefix):] for n in f.keys() if n.startswith(prefix)}
  assert names==set(target),(len(names),len(target),names^set(target))
  with torch.no_grad():
   for n,t in target.items():
    value=f.get_tensor(prefix+n);assert value.shape==t.shape
    t.copy_(value)
 return len(target)
def build(source,*,weights_required,override_cfg=None):
 assert weights_required and override_cfg is not None,'This harness supports fresh controlled training only'
 src=Path(source).resolve();assert src.parent==E/'templates',src
 resolved,model,cfg=original_build(str(src),weights_required=False,override_cfg=override_cfg)
 seed=int(override_cfg.project.seed);route=src.name
 model.freeze_modules(list(override_cfg.model.freeze))
 shared=load_file(str(E/'initialization'/f'common-seed{seed}.safetensors'))
 names={n for n,p in model.named_parameters() if p.requires_grad};assert names==set(shared),(names^set(shared))
 missing,unexpected=model.load_state_dict(shared,strict=False);assert not unexpected and not(names&set(missing))
 assert all(torch.equal(model.state_dict()[n],v) for n,v in shared.items())
 del shared
 text_count=load_prefix(model.video_backbone.text_encoder,R/'assets/dinov3-policy/checkpoint_step_9890.safetensors','video_backbone.text_encoder.')
 enc=model.video_backbone.video_encoder
 if route in ['pca','svae']:
  load_prefix(enc._m,R/'assets/dinov3-study/encoder.safetensors','video_backbone.video_encoder._m.')
  if route=='pca':
   from openwam.model.video_backbone.encoder.dinov3_pca import FrozenPCA
   enc._pca.load_state_dict(FrozenPCA.from_file(R/'results/layers-20260922/L12/pca.pt').state_dict(),strict=True)
  else:
   from openwam.model.video_backbone.encoder.svae.model import load_svae
   fitted=load_svae(str(R/'results/layers-20260922/L12/seed42/final.pt'))
   enc._svae.load_state_dict(fitted.state_dict(),strict=True);del fitted
 elif route=='wan':
  from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder
  model.video_backbone.video_encoder=WanVideoVAEEncoder.from_pretrained(str(R/'assets/wan22-vae-baseline'))
 else:raise ValueError(route)
 model.freeze_modules(list(override_cfg.model.freeze))
 assert {n for n,p in model.named_parameters() if p.requires_grad}==names
 assert not any(p.requires_grad for p in model.video_backbone.video_encoder.parameters())
 record('initialization.json',{'route':route,'seed':seed,'common_sha256':sha(E/'initialization'/f'common-seed{seed}.safetensors'),'equal_common_tensors':len(names),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'frozen_text_tensors':text_count,'published_world_action_weights_loaded':False})
 gc.collect();return resolved,model,cfg
loader.build_architecture_from_ckpt_dir=build
from openwam.train.openwam_trainer import OpenWAMTrainer
original_prepare=OpenWAMTrainer.prepare_accelerate
original_log=OpenWAMTrainer.log_step
original_loss=OpenWAMTrainer.compute_loss

def prepare(self,*args,**kwargs):
 result=original_prepare(self,*args,**kwargs)
 arch=self.accelerator.unwrap_model(self.architecture);enc=arch.video_backbone.video_encoder
 route=Path(self.cfg.training.finetune_ckpt_path).name;template=E/'templates'/route
 entry=OmegaConf.to_container(self.cfg.model.video_backbone.components,resolve=True)[1]
 # Actual encoder sidecar construction and strict reload, then the same native path.
 other=type(enc).from_skeleton(entry,encoder_cfg=self.cfg.model.video_backbone.encoder,ckpt_dir=str(template))
 other.load_state_dict({n:v.detach().cpu() for n,v in enc.state_dict().items()},strict=True)
 other.to(device=self.accelerator.device,dtype=torch.bfloat16).eval();enc.eval()
 rng=np.random.default_rng(991);frames=[Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)) for _ in range(9)]
 with torch.no_grad():
  x=enc.preprocess_video(frames);a=enc.batch_encode(x);b=other.batch_encode(other.preprocess_video(frames))
  assert a.shape==(1,48,3,4,4),a.shape
  assert torch.isfinite(a).all() and torch.isfinite(b).all()
  delta=float((a.float()-b.float()).abs().max());assert delta<1e-4,delta
  first=enc.batch_encode(enc.preprocess_video(frames[:1]));causal=float((a[:,:,:1].float()-first.float()).abs().max())
  assert causal<.02,causal
 record('encoder-parity.json',{'passed':True,'shape':list(a.shape),'strict_reload_max_abs':delta,'first_frame_max_abs':causal,'actual_gpu':torch.cuda.get_device_name(),'input':'fixed synthetic9frames64x64; no heldout scenes'})
 del other,a,b,x,first;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 return result

def loss(self,batch):
 step=getattr(self,'_study_microstep',0);self._study_microstep=step+1
 rows=batch if isinstance(batch,list) else [batch]
 rng=np.random.default_rng(np.random.SeedSequence([int(self.cfg.project.seed),int(os.environ.get('RANK',0)),step,4817]))
 out=[]
 for row in rows:
  sigma=float(rng.choice([0.,.04,.10],p=[.5,.25,.25]));row=dict(row)
  if sigma:
   row['video']=[Image.fromarray(np.rint(np.clip(np.asarray(im,dtype=np.float32)+rng.normal(0,255*sigma,np.asarray(im).shape),0,255)).astype(np.uint8)) for im in row['video']]
   row['first_frame_image']=[row['video'][0]]
  out.append(row)
 return original_loss(self,out)
def log(self,**kw):
 row={k:kw[k] for k in ['global_step','opt_step','steps_per_sec','batch_size','lr']}
 row.update(metrics={k:float(v) for k,v in kw['metrics'].items()},allocated_bytes=torch.cuda.max_memory_allocated(),wall_time=time.time(),rank=int(os.environ.get('RANK',0)))
 assert all(math.isfinite(x) for x in row['metrics'].values()),row
 with (RUN/f'metrics-rank{row["rank"]}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 for pause in [Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json'),Path('/home/zifanz4/openwam-experiments/studies/endtoend-20260923/paused.json')]:
  if pause.exists():raise RuntimeError('User pause marker present; stopping this owned job')
 return original_log(self,**kw)
OpenWAMTrainer.prepare_accelerate=prepare;OpenWAMTrainer.compute_loss=loss;OpenWAMTrainer.log_step=log
sys.path.insert(0,str(E/'OpenWAM/scripts'))
import train
if __name__=='__main__':train.main()
