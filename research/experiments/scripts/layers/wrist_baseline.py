"""Fixed wrist-camera readout baseline; original train split only, no scoring."""
import json,os,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
import common as c
from readout_v2 import visual,select
R=Path(os.environ['LAYER_RUN']);O=R/'wrist-baseline'
def main():
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    assert not (O/'started.json').exists(),'One bounded attempt; do not rerun'
    O.mkdir(exist_ok=True);c.write(O/'started.json',{'started_at':__import__('datetime').datetime.now().astimezone().isoformat(),'fresh_scored':False})
    begin=time.monotonic();original=json.loads((R/'readout/manifest.json').read_text());indices=np.array([i for i,r in enumerate(original) if r['split']=='train']);rows=[dict(original[i]) for i in indices];d=np.load(R/'readout/vectors.npz');state=d['state'][indices];target=d['target'][indices];assert len(rows)==1260
    tasks=np.array([r['task'] for r in rows]);eps=np.array([r['episode'] for r in rows]);features={s:[] for s in c.SOURCES};enc=c.encoder();last=None;f=None
    try:
        for i,row in enumerate(rows):
            p=c.ROOT/row['path']
            if p!=last:
                if f is not None:f.close()
                f=h5py.File(p,'r');last=p
            raw=bytes(f['observation/left_camera/rgb'][row['frame']]);im=c.resize_head(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR));z=c.encode(enc,enc.preprocess_video([im]*9));row['camera']='left_camera';row['current_image_sha256']=__import__('hashlib').sha256(raw).hexdigest()
            for source in c.SOURCES:features[source].append(z[source][0,:,:1].cpu().half())
            if (i+1)%120==0:c.write(O/'progress.json',{'stage':'extract','frames':i+1,'elapsed_seconds':time.monotonic()-begin});print('WRIST_EXTRACT',i+1,flush=True)
    finally:
        if f is not None:f.close()
    del enc;torch.cuda.empty_cache();arrays={}
    for source in c.SOURCES:arrays.update(visual(torch.stack(features[source]),source));del features[source]
    np.savez(O/'train-vectors.npz',**arrays,state=state,target=target);c.write(O/'manifest.json',rows)
    coefficients={};choices={};states={};scales={};metadata={};names=[s+'_'+v for s in c.SOURCES for v in ['raw','pca','svae42','svae43','svae44']]
    for task in c.TASKS:
        tr=tasks==task;assert int(tr.sum())==420
        for name in names:
            key=task+'/'+name;x=np.concatenate([arrays[name],state],1)
            decisions=[]
            for gi,sl in enumerate([slice(0,6),slice(6,8)]):
                m,dec=select(x[tr].astype(float),target[tr,sl].astype(float),eps[tr]);decisions.append(dec)
                for k,v in m.items():coefficients[f'{key}/{gi}/{k}']=v
            choices[key]=decisions
            xm=x[tr].mean(0);xs=x[tr].std(0).clip(1e-6);ym=target[tr].mean(0);ys=target[tr].std(0).clip(1e-6);xx=torch.from_numpy((x[tr]-xm)/xs).cuda();yy=torch.from_numpy((target[tr]-ym)/ys).cuda()
            torch.manual_seed(42);model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,8)).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);rng=torch.Generator().manual_seed(42)
            for step in range(500):
                batch=torch.randint(len(xx),(64,),generator=rng).cuda();optimizer.zero_grad(set_to_none=True);err=(model(xx[batch])-yy[batch]).square();loss=.5*err[:,:6].mean()+.5*err[:,6:].mean();assert torch.isfinite(loss);loss.backward();optimizer.step()
            model.cpu();states[key]={k:v.detach().clone() for k,v in model.state_dict().items()};scales[key]=[torch.from_numpy(a.copy()) for a in [xm,xs,ym,ys]];metadata[key]={'steps':500,'seed':42,'final_training_batch_loss':float(loss),'parameters':sum(p.numel() for p in model.parameters())};del optimizer,model,xx,yy;torch.cuda.empty_cache();c.write(O/'progress.json',{'stage':'fit','readouts_finished':len(states),'elapsed_seconds':time.monotonic()-begin});print('WRIST_FIT',key,flush=True)
    np.savez(O/'coefficients.npz',**coefficients);torch.save({'states':states,'scales':scales,'metadata':metadata},O/'models.pt');c.write(O/'selection.json',choices);c.write(O/'training.json',metadata)
    c.write(O/'complete.json',{'completed_at':__import__('datetime').datetime.now().astimezone().isoformat(),'elapsed_seconds':time.monotonic()-begin,'training_episodes':105,'frames':1260,'paired_group_ridge_models':45,'mlp_models':45,'fresh_scored':False,'sha256':{name:c.sha(O/name) for name in ['coefficients.npz','models.pt','selection.json','training.json','manifest.json']}})
if __name__=='__main__':main()
