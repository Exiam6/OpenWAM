"""Fit matched-codec readouts on original training data only; no scoring."""
import datetime,gc,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ['WAM_ROOT']);L=ROOT/'results/layers-20260922';O=ROOT/'results/goalaux-20260922';OUT=O/'readouts'
os.environ['LAYER_RUN']=str(L);sys.path.insert(0,str(ROOT/'scripts/layers'))
import common as c
from readout_v2 import select,pool
TASKS=c.TASKS;NAMES=[f'{arm}_svae{seed}' for arm in ['reconstruction_only','goal_auxiliary'] for seed in [42,43,44]]

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
 assert not (OUT/'started.json').exists();p=json.loads((OUT/'protocol.json').read_text())
 assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(p['deadline'])
 for file,expected in p['sha256'].items():assert c.sha(ROOT/file)==expected,file
 assert json.loads((O/'training-audit.json').read_text())['passed']
 c.write(OUT/'started.json',{'started_at':datetime.datetime.now().astimezone().isoformat(),'protocol_sha256':c.sha(OUT/'protocol.json')});start=time.monotonic()
 inputs=json.loads((O/'training-inputs.json').read_text());labels=json.loads((L/'initial-goal-training-labels.json').read_text());lookup={(r['task'],r['episode']):r for r in labels['records']};features=[];rows=[];states=[];targets=[];goals=[]
 for task in TASKS:
  episodes=sorted({r['episode'] for r in inputs['rows'] if r['task']==task});assert len(episodes)==35
  for episode in episodes:
   path=L/'features'/task/f'episode{episode}.pt';assert c.sha(path)==inputs['sha256'][str(path.relative_to(ROOT))]
   d=torch.load(path,map_location='cpu',weights_only=False);assert all(r['split']=='train' and r['task']==task and r['episode']==episode for r in d['manifest']);features.append(d['features']['L12'][:,:,:1].clone());rows.extend(d['manifest']);states.append(d['state']);targets.append(d['target']);assert lookup[task,episode]['valid'];goals.extend([lookup[task,episode]['native_endpose_position'][:2]]*len(d['manifest']))
 assert rows==inputs['rows'];x=torch.cat(features);del features;gc.collect()
 state=torch.cat(states).numpy().astype(np.float32);target=torch.cat(targets).numpy().astype(np.float32);goal=np.array(goals,dtype=np.float32);tasks=np.array([r['task'] for r in rows]);episodes=np.array([r['episode'] for r in rows]);initial=np.array([r['frame']==0 for r in rows]);assert initial.sum()==105
 vectors={}
 for name in NAMES:
  arm,seed=name.rsplit('_svae',1);ck=torch.load(O/'models'/arm/f'seed{seed}/final.pt',map_location='cpu',weights_only=False);m=c.SVAE(**ck['model_config']).cuda();m.load_state_dict(ck['state_dict'],strict=True);m.eval().requires_grad_(False);parts=[]
  with torch.inference_mode():
   for batch in x.split(8):
    with torch.autocast('cuda',dtype=torch.bfloat16):z=m.encode_mean(batch.cuda().float())
    parts.append(pool(z))
  vectors[name]=np.concatenate(parts);assert vectors[name].shape==(1260,192) and np.isfinite(vectors[name]).all();del m,parts;gc.collect();torch.cuda.empty_cache();print('TRAIN_VECTORS',name,flush=True)
 np.savez(OUT/'training-vectors.npz',**vectors,state=state,target=target,goal=goal,initial=initial);c.write(OUT/'training-manifest.json',rows)
 coefficients={};selections={};models={};scales={};metadata={}
 for task in TASKS:
  for name in NAMES:
   for endpoint in ['state','goal']:
    use=(tasks==task)&(initial if endpoint=='goal' else True);xx=vectors[name] if endpoint=='goal' else np.concatenate([vectors[name],state],1);yy=goal if endpoint=='goal' else target;xx=xx[use];yy=yy[use];eps=episodes[use];key=f'{endpoint}/{task}/{name}'
    groups=[(0,slice(0,6)),(1,slice(6,8))] if endpoint=='state' else [(0,slice(0,2))]
    for group,sl in groups:
     fitted,selection=select(xx.astype(float),yy[:,sl].astype(float),eps);selections[f'{key}/{group}']=selection
     for field,value in fitted.items():coefficients[f'{key}/{group}/{field}']=value
    xm=xx.mean(0);xs=xx.std(0).clip(1e-6);ym=yy.mean(0);ys=yy.std(0).clip(1e-6);trainx=torch.from_numpy((xx-xm)/xs).cuda();trainy=torch.from_numpy((yy-ym)/ys).cuda();torch.manual_seed(42)
    m=torch.nn.Sequential(torch.nn.Linear(xx.shape[1],128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,yy.shape[1])).cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);rng=torch.Generator().manual_seed(42)
    for step in range(500):
     idx=torch.randint(len(xx),(64,),generator=rng).cuda();opt.zero_grad(set_to_none=True);err=(m(trainx[idx])-trainy[idx]).square();loss=.5*err[:,:6].mean()+.5*err[:,6:].mean() if endpoint=='state' else err.mean();assert torch.isfinite(loss);loss.backward();opt.step()
    models[key]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};scales[key]=[torch.from_numpy(a.copy()) for a in [xm,xs,ym,ys]];metadata[key]={'train_examples':len(xx),'train_episodes':35,'seed':42,'steps':500,'last_training_batch_loss':float(loss.detach()),'output_dimensions':yy.shape[1]}
    del m,opt,trainx,trainy;torch.cuda.empty_cache();print('READOUT_FIT',key,flush=True)
 np.savez(OUT/'coefficients.npz',**coefficients);torch.save({'states':models,'scales':scales,'metadata':metadata},OUT/'mlp.pt');c.write(OUT/'selection.json',selections);c.write(OUT/'training.json',metadata)
 anchor_files=['readout/coefficients.npz','readout/mlp/models.pt','initial-goal/coefficients.npz','initial-goal/models.pt','initial-goal/constants.json']
 anchors={'source':'completed original layer study; no anchor fit repeated','names':['proprio','L12_raw','L12_pca','L12_svae42','L12_svae43','L12_svae44'],'sha256':{str((L/f).relative_to(ROOT)):c.sha(L/f) for f in anchor_files}};c.write(OUT/'anchors.json',anchors)
 files=['training-vectors.npz','training-manifest.json','coefficients.npz','mlp.pt','selection.json','training.json','anchors.json']
 c.write(OUT/'complete.json',{'completed_at':datetime.datetime.now().astimezone().isoformat(),'wall_seconds':time.monotonic()-start,'reducer_models':6,'ridge_endpoint_fits':36,'ridge_group_fits':54,'mlp_fits':36,'training_episodes':105,'fresh_scored':False,'development_scored':False,'sha256':{f:c.sha(OUT/f) for f in files}})
 print('TRAIN_ONLY_READOUTS_COMPLETE',flush=True)
if __name__=='__main__':main()
