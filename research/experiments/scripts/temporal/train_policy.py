"""Matched continuation of native S-VAE policy; only temporal kernel differs."""
import datetime,gc,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
from omegaconf import OmegaConf
from openwam.train.utils import ckpt_model_loader as loader
from openwam.model.video_backbone.encoder.dinov3_temporal import TemporalDinoV3VideoEncoder
R=Path('/data02/zifanz4/openwam-experiments');N=R/'temporal-20260926';E=R/'endtoend-20260923';P=Path('/home/zifanz4/openwam-experiments');S=P/'studies/temporal-20260926';RUN=Path(os.environ['OPENWAM_RUN_DIR']);ARM=RUN.name.split('-seed')[0];assert ARM in ['mean','learned']
def write(name,x):(RUN/name).write_text(json.dumps(x,indent=2)+'\n')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
original_build=loader.build_architecture_from_ckpt_dir

def build(source,*,weights_required,override_cfg=None):
 assert weights_required and override_cfg is not None
 assert override_cfg.model.video_backbone.encoder.name=='dinov3'
 seed=int(override_cfg.project.seed);assert seed in [42,43,44]
 src=Path(source);assert src.parent.parent==E/'training'/f'svae-seed{seed}',src
 checkpoint=src/'checkpoint_step_12000.safetensors';digest=sha(checkpoint)
 expected=json.loads((P/'studies/endtoend-20260923/evaluation-ready.json').read_text())['checkpoint_sha256'][f'svae-seed{seed}'];assert digest==expected
 resolved,model,cfg=original_build(str(src),weights_required=True,override_cfg=override_cfg)
 old=model.video_backbone.video_encoder
 enc=TemporalDinoV3VideoEncoder(old._m,embed_dim=old._raw_embed_dim,patch_size=old.properties.spatial_compression,num_register_tokens=old._num_register_tokens,svae_config=old._svae.config_dict(),svae_target_dim=48)
 missing,unexpected=enc.load_state_dict(old.state_dict(),strict=False);assert missing==['_temporal.logits'] and not unexpected
 if ARM=='learned':
  complete=json.loads((N/'fit/complete.json').read_text());assert sha(N/'fit/temporal-final.pt')==complete['pool_sha256']
  enc._temporal.load_state_dict(torch.load(N/'fit/temporal-final.pt',map_location='cpu',weights_only=True)['pool_state_dict'],strict=True)
 model.video_backbone.video_encoder=enc;override_cfg.model.video_backbone.encoder.name='dinov3_temporal';model.freeze_modules(list(override_cfg.model.freeze))
 assert not any(p.requires_grad for p in enc.parameters());count=sum(p.numel() for p in model.parameters() if p.requires_grad);assert count==295035604,count
 write('initialization.json',{'time':datetime.datetime.now().astimezone().isoformat(),'arm':ARM,'seed':seed,'source':str(checkpoint),'source_sha256':digest,'original_full_strict_load':True,'encoder_missing_only_new_kernel':missing,'frozen_kernel_parameters':enc._temporal.logits.numel(),'trainable_parameters':count,'encoder_state_sha256':hashlib.sha256(enc._temporal.logits.detach().cpu().numpy().tobytes()).hexdigest()})
 del old;gc.collect();return resolved,model,cfg
loader.build_architecture_from_ckpt_dir=build
from openwam.train.openwam_trainer import OpenWAMTrainer
original_prepare=OpenWAMTrainer.prepare_accelerate;original_setup=OpenWAMTrainer.setup_output_dir;original_loss=OpenWAMTrainer.compute_loss;original_log=OpenWAMTrainer.log_step

def setup(self,*a,**kw):
 result=original_setup(self,*a,**kw);self._temporal_saved_dir=result[0];return result

def prepare(self,*a,**kw):
 result=original_prepare(self,*a,**kw);enc=self.accelerator.unwrap_model(self.architecture).video_backbone.video_encoder;enc.eval()
 entry=OmegaConf.to_container(self.cfg.model.video_backbone.components,resolve=True)[1]
 other=TemporalDinoV3VideoEncoder.from_skeleton(entry,encoder_cfg=self.cfg.model.video_backbone.encoder,ckpt_dir=self._temporal_saved_dir)
 other.load_state_dict({k:v.detach().cpu() for k,v in enc.state_dict().items()},strict=True);other.to(device=self.accelerator.device,dtype=torch.bfloat16).eval()
 rng=np.random.default_rng(426);frames=[Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)) for _ in range(9)]
 with torch.no_grad():
  x=enc.preprocess_video(frames);z=enc.batch_encode(x);reloaded=other.batch_encode(other.preprocess_video(frames));assert z.shape==(1,48,3,4,4) and torch.isfinite(z).all()
  delta=float((z.float()-reloaded.float()).abs().max());assert delta<1e-4,delta
  changed=x.clone();changed[:,:,5:]=changed[:,:,5:].flip(-1);causal=float((z[:,:,:2].float()-enc.batch_encode(changed)[:,:,:2].float()).abs().max());assert causal<1e-4,causal
 write('encoder-parity.json',{'passed':True,'native_save_reload_max_abs':delta,'later_group_causality_max_abs':causal,'shape':list(z.shape),'saved_sidecar_dir':self._temporal_saved_dir,'input':'synthetic9frames64x64; no test data'})
 del other,z,reloaded,x;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();return result

def loss(self,batch):
 step=getattr(self,'_new_microstep',0);self._new_microstep=step+1;rows=batch if isinstance(batch,list) else [batch];rng=np.random.default_rng(np.random.SeedSequence([int(self.cfg.project.seed),int(os.environ.get('RANK',0)),step,9826]));out=[];provenance=[]
 for row in rows:
  row=dict(row);sigma=float(rng.choice([0.,.04,.10],p=[.5,.25,.25]))
  if sigma:
   row['video']=[Image.fromarray(np.rint(np.clip(np.asarray(im,dtype=np.float32)+rng.normal(0,255*sigma,np.asarray(im).shape),0,255)).astype(np.uint8)) for im in row['video']];row['first_frame_image']=[row['video'][0]]
  provenance.append({'task':row['task_name'],'episode':Path(row['episode_path']).name,'frame':int(row['start_frame']),'sigma':sigma,'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest()});out.append(row)
 with (RUN/'batch-stream.jsonl').open('a') as f:f.write(json.dumps({'microstep':step,'rows':provenance})+'\n')
 return original_loss(self,out)

def log(self,**kw):
 row={k:kw[k] for k in ['global_step','opt_step','steps_per_sec','batch_size','lr']};row.update(metrics={k:float(v) for k,v in kw['metrics'].items()},time=datetime.datetime.now().astimezone().isoformat(),peak_cuda_bytes=torch.cuda.max_memory_allocated())
 assert all(math.isfinite(v) for v in row['metrics'].values())
 with (RUN/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(json.loads((S/'protocol.json').read_text())['deadline'])
 for p in [S/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),p
 return original_log(self,**kw)
OpenWAMTrainer.setup_output_dir=setup;OpenWAMTrainer.prepare_accelerate=prepare;OpenWAMTrainer.compute_loss=loss;OpenWAMTrainer.log_step=log
sys.path.insert(0,str(N/'OpenWAM/scripts'))
import train
if __name__=='__main__':train.main()
