"""Train-only dataset stats and common native model initialization (no evaluation)."""
import argparse, copy, gc, hashlib, json, os, shutil, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
R=Path('/data02/zifanz4/openwam-experiments'); E=R/'endtoend-20260923'; S=Path('/home/zifanz4/openwam-experiments/studies/endtoend-20260923')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def setup():
 E.mkdir(exist_ok=True);assert shutil.disk_usage(E).free>50*2**30
 source=R/'results/layers-20260922/native-preflight/train-data'
 files=sorted(source.glob('*/aloha-agilex_clean_50/data/episode*.hdf5'));assert len(files)==105,len(files)
 expected=set(np.random.default_rng(42).permutation(50)[:35].tolist())
 tasks=sorted(set(f.parents[2].name for f in files));assert tasks==['adjust_bottle','handover_block','place_object_basket']
 for task in tasks:assert {int(f.stem[7:]) for f in files if f.parents[2].name==task}==expected
 if not (E/'data-manifest.json').exists():dump(E/'data-manifest.json',{'created_at':datetime.now().astimezone().isoformat(),'source':str(source),'split_seed':42,'train_only':True,'files':[{'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]})
 stats=E/'normalization_stats.npy'
 if not stats.exists():
  from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import compute_multitask_robotwin_stats,atomic_save_stats_npy
  result=compute_multitask_robotwin_stats(str(source),'aloha-agilex',variant='clean_50',tasks=tasks)
  assert result['eef']['mean'].shape==(20,);atomic_save_stats_npy(str(stats),result)
 template=OmegaConf.to_container(OmegaConf.load(R/'results/layers-20260922/native-preflight/trial2/profile.yaml'),resolve=True)
 vb=template['model']['video_backbone'];kw=vb['components'][0]['extra_kwargs']
 kw.update(dim=1024,ffn_dim=4096,num_heads=16,num_layers=12)
 template['model']['action_backbone'].update(dim=512,ffn_dim=2048)
 template['dataloader']['normalization_stats_path']=str(stats)
 template['training'].update(batch_size=2,gradient_accumulation_steps=4,max_steps=64,save_steps=64,offload_optimizer_device='none',zero_stage=2,dataset_num_workers=2,keep_last_k_ckpts=1)
 for route in ['svae','pca','wan']:
  dest=E/'templates'/route;dest.mkdir(parents=True,exist_ok=True);cfg=copy.deepcopy(template)
  enc={'name':'wan22_vae','model_path':str(R/'assets/wan22-vae-baseline')}
  if route!='wan':
   enc={'name':'dinov3' if route=='svae' else 'dinov3_pca','model_path':str(R/'assets/dinov3-study/dinov3')}
   shutil.copytree(R/'assets/dinov3-study/dinov3',dest/'dinov3',dirs_exist_ok=True)
   if route=='svae':
    enc.update(svae_path=str(R/'results/layers-20260922/L12/seed42/final.pt'),svae_target_dim=48)
    sv=torch.load(enc['svae_path'],map_location='cpu',weights_only=False)
    dump(dest/'svae_config.json',{'format_version':sv['format_version'],'model_config':sv['model_config']})
   else:
    enc['pca_path']=str(R/'results/layers-20260922/L12/pca.pt')
    dump(dest/'pca_config.json',{'format_version':1,'input_dim':768,'latent_dim':48,'whiten':True})
  cfg['model']['video_backbone']['encoder']=enc
  # Only tokenizer files/config are reused; no released policy weights in common init.
  if not (dest/'tokenizer').exists():os.symlink(R/'assets/dinov3-policy/tokenizer',dest/'tokenizer',target_is_directory=True)
  cfg['training'].update(finetune_ckpt_path=str(dest),output_path=str(E/'profiles'/route/'output'))
  cfg['project'].update(output_dir=str(E/'profiles'/route/'hydra'),seed=42)
  cfg['hydra']['run']['dir']=cfg['project']['output_dir']
  OmegaConf.save(OmegaConf.create(cfg),dest/'config.yaml')
 dump(S/'data-preparation.json',{'time':datetime.now().astimezone().isoformat(),'episodes':len(files),'tasks':tasks,'manifest':str(E/'data-manifest.json'),'stats':str(stats),'stats_sha256':sha(stats),'data_manifest_sha256':sha(E/'data-manifest.json'),'encoder_fit':'previous frozen105 train scenes; SVAE layer12 reducer seed42 for all policy seeds','scale':'controlled native 12-layer world1024/action512, not published5B'})
 print('SETUP COMPLETE',flush=True)
def common(seed):
 target=E/'initialization'/f'common-seed{seed}.safetensors';assert not target.exists(),target
 torch.manual_seed(seed);torch.set_num_threads(4)
 from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
 from openwam.model.video_backbone.wan.reinit import reinit_dit_from_scratch
 _,model,cfg=build_architecture_from_ckpt_dir(str(E/'templates/svae'),weights_required=False)
 reinit_dit_from_scratch(model.video_backbone,verbose=False)
 model.freeze_modules(list(cfg.model.freeze))
 names={n for n,p in model.named_parameters() if p.requires_grad};state=model.state_dict()
 shared={n:state[n].detach().cpu().contiguous() for n in sorted(names)}
 assert all(n.startswith(('video_backbone.dit.','action_backbone.','proprio_encoder.')) for n in names)
 assert all(torch.isfinite(t).all() for t in shared.values())
 target.parent.mkdir(exist_ok=True);save_file(shared,str(target))
 dump(target.with_suffix('.json'),{'seed':seed,'shared_trainable_parameters':sum(t.numel() for t in shared.values()),'tensors':len(shared),'sha256':sha(target),'frozen_weights_excluded':True,'shapes':{n:list(t.shape) for n,t in shared.items()}})
 print('COMMON INIT COMPLETE',seed,target,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['setup','common']);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 setup() if a.stage=='setup' else common(a.seed)
