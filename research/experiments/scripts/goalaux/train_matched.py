"""Prospective two-arm/three-seed fixed-budget SVAE continuation; train only."""
import datetime,gc,hashlib,importlib.util,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from objective_aligned import waypoint_objective,with_waypoint_objective

ROOT=Path(os.environ['WAM_ROOT']);L=ROOT/'results/layers-20260922';O=ROOT/'results/goalaux-20260922'
TASKS=['adjust_bottle','handover_block','place_object_basket'];SEEDS=[42,43,44];STEPS=2000

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def main():
 p=json.loads((O/'matched-protocol.json').read_text());assert not (O/'training-started.json').exists()
 assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(p['deadline'])
 assert json.loads((O/'gpu-preflight-v2/result.json').read_text())['passed']
 for path,expected in p['source_sha256'].items():assert sha(ROOT/path)==expected,path
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 assert torch.cuda.device_count()==1
 native_path=ROOT/'OpenWAM/openwam/model/video_backbone/encoder/svae/model.py';spec=importlib.util.spec_from_file_location('native_svae',native_path);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 manifest=json.loads((L/'features/manifest.json').read_text());labels=json.loads((L/'initial-goal-training-labels.json').read_text());assert labels['dev_or_confirmation_read'] is False
 blocks=[];rows=[];targets=[];first=[];train_indices=[];input_hashes={};constants={}
 for task in TASKS:
  allowed=sorted({r['episode'] for r in manifest if r['task']==task and r['split']=='train'});assert len(allowed)==35
  recs=[r for r in labels['records'] if r['task']==task];assert sorted(r['episode'] for r in recs)==allowed and all(r['valid'] for r in recs)
  yy=np.array([r['native_endpose_position'][:2] for r in recs],dtype=np.float64);mean=yy.mean(0);std=yy.std(0).clip(1e-6);constants[task]={'mean':mean.tolist(),'std':std.tolist()};start=len(rows)
  for episode in allowed:
   path=L/'features'/task/f'episode{episode}.pt';d=torch.load(path,map_location='cpu',weights_only=False);input_hashes[str(path.relative_to(ROOT))]=sha(path)
   meta=d['manifest'];assert len(meta)==12 and meta[0]['frame']==0 and all(r['task']==task and r['episode']==episode and r['split']=='train' for r in meta)
   assert d['features']['L12'].shape==(12,768,3,24,20)
   original=[r for r in manifest if r['task']==task and r['episode']==episode];assert meta==original
   first.extend([len(rows)]*12);rows.extend(meta);blocks.append(d['features']['L12'].clone());raw=next(r for r in recs if r['episode']==episode)['native_endpose_position'][:2];targets.extend([((np.array(raw)-mean)/std).tolist()]*12)
  train_indices.append(torch.arange(start,len(rows)))
 assert len(rows)==1260 and len(input_hashes)==105
 input_manifest={'frozen_at':datetime.datetime.now().astimezone().isoformat(),'split':'train_only','episodes':105,'clips':1260,'sha256':input_hashes,'rows':rows,'goal_scaling':constants,'steps':STEPS,'seeds':SEEDS,'source_protocol_sha256':sha(O/'matched-protocol.json')}
 assert not (O/'training-inputs.json').exists();write(O/'training-inputs.json',input_manifest)
 x=torch.cat(blocks).cuda().float();del blocks;gc.collect();torch.cuda.empty_cache();target=torch.tensor(targets,dtype=torch.float32,device='cuda');first=torch.tensor(first,dtype=torch.long,device='cuda');task_index=torch.arange(3,device='cuda')
 assert torch.isfinite(x).all() and torch.isfinite(target).all()
 # Check the new readout-aligned normalization path before any retained fit.
 ck=torch.load(L/'L12/seed42/final.pt',map_location='cpu',weights_only=False);m=native.SVAE(**ck['model_config']).cuda();m.load_state_dict(ck['state_dict']);heads=torch.nn.ModuleList([torch.nn.Linear(192,2) for _ in TASKS]).cuda()
 idx=torch.tensor([int(t[0]) for t in train_indices],device='cuda')
 with torch.autocast('cuda',dtype=torch.bfloat16):aux=waypoint_objective(m.encode_mean(x[idx,:,:1]),heads,task_index,target[idx])
 aux.backward();grad_norm=float(torch.sqrt(sum(q.grad.float().square().sum() for n,q in m.named_parameters() if n.startswith('enc_'))));assert np.isfinite(grad_norm) and grad_norm>0
 assert all(q.grad is None for n,q in m.named_parameters() if n.startswith('dec_'))
 write(O/'aligned-objective-preflight.json',{'passed':True,'encoder_gradient_norm':grad_norm,'deterministic_backward':True,'training_samples':3,'checkpoint_saved':False,'goal_input_matches_readout_per_token_norm_eps':1e-6})
 del m,heads,aux,ck;gc.collect();torch.cuda.empty_cache()
 write(O/'training-started.json',{'started_at':datetime.datetime.now().astimezone().isoformat(),'deadline':p['deadline'],'protocol_sha256':sha(O/'matched-protocol.json'),'input_manifest_sha256':sha(O/'training-inputs.json')})
 completed=[]
 for seed in SEEDS:
  for arm,weight in [('reconstruction_only',0.),('goal_auxiliary',0.1)]:
   dest=O/'models'/arm/f'seed{seed}';dest.mkdir(parents=True);assert not (dest/'final.pt').exists()
   ck=torch.load(L/f'L12/seed{seed}/final.pt',map_location='cpu',weights_only=False)
   torch.manual_seed(seed);model=native.SVAE(**ck['model_config']).cuda();model.load_state_dict(ck['state_dict'],strict=True);model.train();heads=torch.nn.ModuleList([torch.nn.Linear(192,2) for _ in TASKS]).cuda()
   parameters=list(model.parameters())+list(heads.parameters());opt=torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.99),weight_decay=1e-4);sampler=torch.Generator().manual_seed(seed);torch.manual_seed(seed);sample_hash=hashlib.sha256();history=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
   for step in range(1,STEPS+1):
    assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(p['deadline'])
    selected=torch.stack([ii[torch.randint(len(ii),(1,),generator=sampler)[0]] for ii in train_indices]);sample_hash.update(selected.numpy().tobytes());selected=selected.cuda();opt.zero_grad(set_to_none=True)
    lr=1e-4*min(1.,step/20)*(.95+.05*np.cos(np.pi*step/STEPS))
    for group in opt.param_groups:group['lr']=lr
    with torch.autocast('cuda',dtype=torch.bfloat16):
     output=model(x[selected]);base,parts=native.svae_loss(output,beta=1e-4,cos_weight=1.)
     if weight:
      mu=model.encode_mean(x[first[selected],:,:1]);aux=waypoint_objective(mu,heads,task_index,target[selected]);loss=base+weight*aux
     else:loss=with_waypoint_objective(base,None,None,None,None,0.)
    assert torch.isfinite(loss),step
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm);opt.step()
    if step in [1,20,50,100] or step%200==0:
     row={'step':step,'loss':float(loss.detach()),'native_loss':float(base.detach()),'auxiliary_loss':float(aux.detach()) if weight else None,'elapsed_seconds':time.monotonic()-start,'gradient_norm_before_clip':float(norm)};history.append(row);write(dest/'curve.json',history);print(arm,seed,json.dumps(row),flush=True)
   final={'format_version':2,'model_config':model.config_dict(),'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'seed':seed,'source':'L12','arm':arm,'step':STEPS,'auxiliary_weight':weight,'training_input_manifest_sha256':sha(O/'training-inputs.json')};torch.save(final,dest/'final.pt');torch.save({k:v.detach().cpu().clone() for k,v in heads.state_dict().items()},dest/'training-only-head.pt')
   info={'seed':seed,'arm':arm,'steps':STEPS,'wall_seconds':time.monotonic()-start,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,'checkpoint_sha256':sha(dest/'final.pt'),'sample_sequence_sha256':sample_hash.hexdigest(),'selection':'fixed final step; no development/test scores','initial_checkpoint_sha256':sha(L/f'L12/seed{seed}/final.pt')};write(dest/'complete.json',info);completed.append(info);write(O/'training-progress.json',{'completed':completed,'training_only':True})
   del model,heads,opt,output,base,loss,parameters,final,ck
   if weight:del mu,aux
   gc.collect();torch.cuda.empty_cache()
  assert completed[-1]['sample_sequence_sha256']==completed[-2]['sample_sequence_sha256']
 write(O/'training-complete.json',{'completed_at':datetime.datetime.now().astimezone().isoformat(),'models':completed,'matched_sample_sequences':True,'readouts_fit':False,'fresh_scored':False,'policy_adaptation':False})
 print('ALL_SIX_MATCHED_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
