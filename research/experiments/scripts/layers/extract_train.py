"""Shared-task, equal-budget layer x compression study. No fresh test scoring."""
import argparse,gc,json,math,os,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
import common as c
from torch.nn import functional as F
RUN=Path(os.environ['LAYER_RUN']);CACHE=RUN/'features';STEPS=2000

def extract():
    assert json.loads((RUN/'interface-preflight.json').read_text())['passed']
    assert not (CACHE/'complete.json').exists(),'Do not rerun completed extraction'
    CACHE.mkdir(parents=True,exist_ok=True);enc=c.encoder();manifest=[];begin=time.monotonic();order=np.random.default_rng(42).permutation(50)
    chosen={int(e):'train' if i<35 else 'development' for i,e in enumerate(order[:40])}
    for ti,task in enumerate(c.TASKS):
        paths=sorted((c.ROOT/'data'/task).rglob('*.hdf5'),key=lambda p:int(p.stem.replace('episode','')))
        for ep,path in enumerate(paths):
            if ep not in chosen:continue
            dest=CACHE/task/f'episode{ep}.pt';assert not dest.exists(),dest
            features={s:[] for s in c.SOURCES};meta=[];states=[];targets=[]
            with h5py.File(path,'r') as f:
                n=len(f['endpose/left_endpose']);starts=np.unique(np.linspace(0,n-33,12,dtype=int));assert len(starts)==12
                for st in starts:
                    indices=list(range(int(st),int(st)+33,4));images=[];hashes=[]
                    for j in indices:
                        raw=bytes(f['observation/head_camera/rgb'][j]);hashes.append(__import__('hashlib').sha256(raw).hexdigest())
                        images.append(c.resize_head(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)))
                    z=c.encode(enc,enc.preprocess_video(images))
                    for source in c.SOURCES:features[source].append(z[source][0].cpu().half())
                    lp=f['endpose/left_endpose'][st];rp=f['endpose/right_endpose'][st]
                    states.append(np.concatenate([lp,rp,[f['endpose/left_gripper'][st],f['endpose/right_gripper'][st]]]))
                    targets.append(np.concatenate([f['endpose/left_endpose'][st+4,:3]-lp[:3],f['endpose/right_endpose'][st+4,:3]-rp[:3],[f['endpose/left_gripper'][st+4],f['endpose/right_gripper'][st+4]]]))
                    meta.append({'task':task,'episode':ep,'episode_uid':ti*1000+ep,'frame':int(st),'frame_indices':indices,'split':chosen[ep],'encoded_head_hashes':hashes,'path':str(path.relative_to(c.ROOT))})
            dest.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'features':{s:torch.stack(x) for s,x in features.items()},'manifest':meta,'state':torch.tensor(np.array(states),dtype=torch.float32),'target':torch.tensor(np.array(targets),dtype=torch.float32)},dest)
            manifest.extend(meta);c.write(CACHE/'progress.json',{'episodes':len(manifest)//12,'clips':len(manifest),'elapsed_seconds':time.monotonic()-begin})
            print('EXTRACT',task,ep,chosen[ep],len(manifest),round(time.monotonic()-begin,1),flush=True)
    c.write(CACHE/'manifest.json',manifest);c.write(CACHE/'complete.json',{'episodes':120,'clips':len(manifest),'elapsed_seconds':time.monotonic()-begin,'cache_bytes':sum(p.stat().st_size for p in CACHE.rglob('*.pt'))})
    del enc;gc.collect();torch.cuda.empty_cache()

def load_source(source):
    xs=[];rows=[];state=[];target=[]
    for task in c.TASKS:
        for path in sorted((CACHE/task).glob('*.pt'),key=lambda p:int(p.stem.replace('episode',''))):
            d=torch.load(path,map_location='cpu',weights_only=False);xs.append(d['features'][source]);rows.extend(d['manifest']);state.append(d['state']);target.append(d['target'])
    return torch.cat(xs),rows,torch.cat(state),torch.cat(target)

def stats(x,ii):
    n=0;su=torch.zeros(768,device='cuda',dtype=torch.float64);ss=su.clone()
    for part in ii.split(3):
        z=x[part].cuda().permute(0,2,3,4,1).reshape(-1,768).double();n+=len(z);su+=z.sum(0);ss+=(z*z).sum(0)
    mu=(su/n).float();std=(ss/n-(su/n)**2).clamp_min(1e-12).sqrt().float();cov=torch.zeros(768,768,device='cuda')
    for part in ii.split(3):
        z=x[part].cuda().permute(0,2,3,4,1).reshape(-1,768).float();z=(z-mu)/std;cov+=z.T@z
    ev,basis=torch.linalg.eigh((cov/n).double().cpu())
    return {'mean':mu.cpu(),'std':std.cpu(),'basis':basis.flip(1).float()[:,:48],'eigenvalues':ev.flip(0).float(),'train_tokens':n}

@torch.inference_mode()
def evaluate_reconstruction(m,x,ii):
    m.eval();su=0.;n=0
    for part in ii.split(3):
        with torch.autocast('cuda',dtype=torch.bfloat16):o=m(x[part].cuda().float())
        e=(o['recon'].float()-o['target'].float()).square();su+=float(e.sum());n+=e.numel()
    return su/n

def fit(source):
    folder=RUN/source;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'training-complete.json').exists(),'No completed-fit replay'
    x,rows,state,target=load_source(source)
    train=[torch.tensor([i for i,r in enumerate(rows) if r['task']==task and r['split']=='train']) for task in c.TASKS]
    ii=torch.cat(train);vv=torch.tensor([i for i,r in enumerate(rows) if r['split']=='development'])
    assert all(len(a)==420 for a in train) and len(vv)==180
    pca=stats(x,ii);torch.save(pca,folder/'pca.pt')
    config=json.loads((c.ROOT/'assets/dinov3-study/svae_config.json').read_text())['model_config']
    for seed in [42,43,44]:
        dest=folder/f'seed{seed}';dest.mkdir(exist_ok=True);assert not (dest/'final.pt').exists()
        torch.manual_seed(seed);m=c.SVAE(**config).cuda();m.set_input_stats(pca['mean'],pca['std'])
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4);rng=torch.Generator().manual_seed(seed)
        history=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for step in range(1,STEPS+1):
            selected=torch.stack([idx[torch.randint(len(idx),(1,),generator=rng)[0]] for idx in train]);m.train();opt.zero_grad(set_to_none=True)
            lr=1e-4*min(1.,step/20)*(.95+.05*np.cos(np.pi*step/STEPS))
            for group in opt.param_groups:group['lr']=lr
            with torch.autocast('cuda',dtype=torch.bfloat16):
                o=m(x[selected].cuda().float());loss,parts=c.svae_loss(o,beta=1e-4*min(1.,step/400),cos_weight=1.)
            assert torch.isfinite(loss),step
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step()
            if step in [20,50,100] or step%200==0:
                row={'step':step,'loss':float(loss),'elapsed_seconds':time.monotonic()-start}
                if step%200==0:row['development_reconstruction_mse']=evaluate_reconstruction(m,x,vv)
                history.append(row);c.write(dest/'curve.json',history);print('TRAIN',source,seed,json.dumps(row),flush=True)
        torch.save({'format_version':2,'model_config':m.config_dict(),'state_dict':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'step':STEPS,'seed':seed,'source':source},dest/'final.pt')
        c.write(dest/'complete.json',{'source':source,'seed':seed,'steps':STEPS,'wall_seconds':time.monotonic()-start,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,'sha256':c.sha(dest/'final.pt'),'selection':'final step; no best-checkpoint selection'})
        del m,opt;gc.collect();torch.cuda.empty_cache()
    c.write(folder/'training-complete.json',{'source':source,'seeds':[42,43,44],'steps':STEPS,'shared_tasks':c.TASKS})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','train']);p.add_argument('--source',choices=c.SOURCES);a=p.parse_args()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(42)
    if a.mode=='extract':extract()
    else:
        assert a.source;fit(a.source)
