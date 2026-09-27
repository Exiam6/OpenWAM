"""Training-only initial-image to expert waypoint readouts; no fresh scoring."""
import datetime,json,os,time
from pathlib import Path
import numpy as np,torch
import common as c
from readout_v2 import select
R=Path(os.environ['LAYER_RUN']);O=R/'initial-goal'
def main():
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    assert not (O/'started.json').exists();O.mkdir(exist_ok=True);c.write(O/'started.json',{'started_at':datetime.datetime.now().astimezone().isoformat()});start=time.monotonic()
    allrows=json.loads((R/'readout/manifest.json').read_text());idx=np.array([i for i,r in enumerate(allrows) if r['split']=='train' and r['frame']==0]);rows=[allrows[i] for i in idx];assert len(rows)==105
    labels=json.loads((R/'initial-goal-training-labels.json').read_text());assert labels['dev_or_confirmation_read'] is False;lookup={(r['task'],r['episode']):r for r in labels['records']};target=np.array([lookup[(r['task'],r['episode'])]['native_endpose_position'][:2] for r in rows],dtype=np.float32);assert all(lookup[(r['task'],r['episode'])]['valid'] for r in rows)
    vectors=np.load(R/'readout/vectors.npz');tasks=np.array([r['task'] for r in rows]);eps=np.array([r['episode'] for r in rows]);names=[s+'_'+v for s in c.SOURCES for v in ['raw','pca','svae42','svae43','svae44']];coef={};choices={};states={};scales={};metadata={};constants={}
    for task in c.TASKS:
        tr=tasks==task;assert tr.sum()==35;constants[task]={'mean':target[tr].astype(float).mean(0).tolist(),'std':target[tr].astype(float).std(0).clip(1e-6).tolist()}
        for name in names:
            key=task+'/'+name;x=vectors[name][idx];m,decision=select(x[tr].astype(float),target[tr].astype(float),eps[tr]);choices[key]=decision
            for k,v in m.items():coef[key+'/'+k]=v
            xm=x[tr].mean(0);xs=x[tr].std(0).clip(1e-6);ym=target[tr].mean(0);ys=target[tr].std(0).clip(1e-6);xx=torch.from_numpy((x[tr]-xm)/xs).cuda();yy=torch.from_numpy((target[tr]-ym)/ys).cuda();torch.manual_seed(42)
            model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,2)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);rng=torch.Generator().manual_seed(42)
            for step in range(500):
                jj=torch.randint(len(xx),(64,),generator=rng).cuda();opt.zero_grad(set_to_none=True);loss=(model(xx[jj])-yy[jj]).square().mean();assert torch.isfinite(loss);loss.backward();opt.step()
            model.cpu();states[key]={k:v.detach().clone() for k,v in model.state_dict().items()};scales[key]=[torch.from_numpy(v.copy()) for v in [xm,xs,ym,ys]];metadata[key]={'steps':500,'seed':42,'final_training_batch_loss':float(loss),'parameters':sum(p.numel() for p in model.parameters())};del model,opt,xx,yy;torch.cuda.empty_cache();print('INITIAL_GOAL_FIT',key,flush=True)
    np.savez(O/'coefficients.npz',**coef);torch.save({'states':states,'scales':scales,'metadata':metadata},O/'models.pt');c.write(O/'selection.json',choices);c.write(O/'training.json',metadata);c.write(O/'constants.json',constants);c.write(O/'training-samples.json',[{'task':r['task'],'episode':r['episode'],'frame':0,'target_xy':y.tolist()} for r,y in zip(rows,target)])
    c.write(O/'complete.json',{'completed_at':datetime.datetime.now().astimezone().isoformat(),'elapsed_seconds':time.monotonic()-start,'ridge_models':45,'mlp_models':45,'episodes':105,'fresh_scored':False,'sha256':{f:c.sha(O/f) for f in ['coefficients.npz','models.pt','selection.json','training.json','constants.json','training-samples.json']}})
if __name__=='__main__':main()
