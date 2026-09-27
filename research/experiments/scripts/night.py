#!/usr/bin/env python3
"""Paired fixed-budget S-VAE experiment. See NIGHT_PROTOCOL.md before interpreting."""
import argparse,gc,json,math,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import h5py,cv2
from PIL import Image
import pilot as p

TASKS=['adjust_bottle','handover_block','place_object_basket']
ROOT=p.ROOT;RUNROOT=ROOT/'results/night-20260921'
SEEDS=[42,43,44];STEPS=2000

def weighted_loss(out,beta,weight):
    loss,stat=p.svae_loss(out,beta=beta,cos_weight=1.)
    if weight==1:return loss,stat
    first={k:v[:,:,:1] for k,v in out.items()}
    first_loss,_=p.svae_loss(first,beta=beta,cos_weight=1.)
    n=out['recon'].shape[2]
    return (n*loss+(weight-1)*first_loss)/(n+weight-1),stat

@torch.inference_mode()
def reconstruction(model,x,indices):
    model.eval();rows=[]
    for part in indices.split(2):
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x[part].cuda().float())
        mse=(out['recon'].float()-out['target'].float()).square().mean((1,3,4))
        rows.extend(mse.cpu().tolist())
    a=np.asarray(rows)
    return {'all_mse':float(a.mean()),'condition_mse':float(a[:,0].mean()),'pooled_target_mse':float(a[:,1:].mean())}

def train(data,pca,seed,weight,folder):
    folder.mkdir(parents=True,exist_ok=True);path=folder/'final.pt'
    if path.exists():return path
    torch.manual_seed(seed);m=p.make_svae(pca);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed);ii=p.ids(data,'train');vv=p.ids(data,'val');x=data['features']
    initial=reconstruction(m,x,vv);history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(1,STEPS+1):
        selected=ii[torch.randint(len(ii),(2,),generator=rng)];z=x[selected].cuda().float();m.train();opt.zero_grad(set_to_none=True)
        lr=1e-4*min(1.,step/20)*(.95+.05*np.cos(np.pi*step/STEPS))
        for group in opt.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=m(z);loss,_=weighted_loss(out,1e-4*min(1.,step/(STEPS*.2)),weight)
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if step%200==0:
            rec={'step':step,'val':reconstruction(m,x,vv),'train_loss':float(loss),'elapsed_seconds':time.perf_counter()-start}
            history.append(rec);p.write_json(folder/'curve.json',history)
            print('TRAIN',folder.name,json.dumps(rec),flush=True)
    torch.save({'format_version':2,'model_config':m.config_dict(),'state_dict':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'step':STEPS,'seed':seed,'condition_weight':weight},path)
    p.write_json(folder/'training.json',{'initial_validation':initial,'final_validation':history[-1]['val'],'seed':seed,'condition_weight':weight,'steps':STEPS,'wall_seconds':time.perf_counter()-start,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9})
    del m,opt;gc.collect();torch.cuda.empty_cache();return path

@torch.inference_mode()
def corruptions(data,folder):
    path=folder/'corruptions.pt'
    if path.exists():return torch.load(path,weights_only=False)
    enc=p.build_encoder();test=p.ids(data,'test').tolist();out={'test_ids':test,'noise10':[],'brightness06':[]};start=time.perf_counter();diff=[]
    for index in test:
        meta=data['manifest'][index]
        with h5py.File(ROOT/'data'/meta['episode_file'],'r') as f:
            images={cam:Image.fromarray(cv2.imdecode(np.frombuffer(bytes(f[f'observation/{cam}/rgb'][meta['start_frame']]),np.uint8),cv2.IMREAD_COLOR)) for cam in p.CAMS}
        frame=p.assemble_multiview_layout(images,p.CAMS,384,320);arr=np.asarray(frame).astype(np.float32)
        rng=np.random.default_rng(10000+meta['episode']*1000+meta['start_frame'])
        images={'clean':frame,'noise10':Image.fromarray(np.clip(arr+rng.normal(0,10,arr.shape),0,255).astype(np.uint8)),'brightness06':Image.fromarray(np.clip(arr*.6,0,255).astype(np.uint8))}
        for name,im in images.items():
            z=enc.batch_encode_pooled_for_svae_training(enc.preprocess_video([im]))[0].cpu().half()
            assert z.shape==(768,1,24,20)
            if name=='clean':diff.append(float((z-data['features'][index,:,:1]).abs().max()))
            else:out[name].append(z)
    for name in ['noise10','brightness06']:out[name]=torch.stack(out[name])
    # Single-frame vs multi-frame GEMMs may round differently in bf16; record, do not silently assert exact equality.
    p.write_json(folder/'corruption-extraction.json',{'wall_seconds':time.perf_counter()-start,'single_frame_vs_clip_max_abs':max(diff),'test_ids':test,'noise_sigma_pixels':10,'brightness_factor':.6,'applied_after_multiview_assembly':True})
    torch.save(out,path);del enc;gc.collect();torch.cuda.empty_cache();return out

def fit_probe(vectors,data):
    tr=p.ids(data,'train').numpy();va=p.ids(data,'val').numpy();y=data['target'].numpy().astype(np.float64)
    x=np.asarray(vectors,dtype=np.float64);xm=x[tr].mean(0);xs=x[tr].std(0).clip(1e-5);ym=y[tr].mean(0);ys=y[tr].std(0).clip(1e-6)
    xx=(x-xm)/xs/np.sqrt(x.shape[1]);ev,q=np.linalg.eigh(xx[tr]@xx[tr].T);ev=ev.clip(0);proj=q.T@((y[tr]-ym)/ys);best=None
    for alpha in [1e-5,1e-4,1e-3,1e-2,.1,1.,10.]:
        coef=q@(proj/(ev[:,None]+alpha));pred=xx[va]@xx[tr].T@coef*ys+ym;score=np.mean(((pred-y[va])/ys)**2)
        if best is None or score<best[0]:best=(score,alpha,coef)
    return {'xm':xm,'xs':xs,'ym':ym,'ys':ys,'train_x':xx[tr],'coef':best[2],'alpha':best[1],'val_mse':best[0],'dim':x.shape[1]}

def predict(probe,x):
    xx=(np.asarray(x,dtype=np.float64)-probe['xm'])/probe['xs']/np.sqrt(probe['dim'])
    return xx@probe['train_x'].T@probe['coef']*probe['ys']+probe['ym']

def vector(lat):
    cur=lat[:,:,0];cur=F.layer_norm(cur.permute(0,2,3,1),(cur.shape[1],),eps=1e-6).permute(0,3,1,2)
    return F.adaptive_avg_pool2d(cur,(2,2)).flatten(1).cpu().numpy()

@torch.inference_mode()
def evaluate(data,pca,corrupt,folder,name,checkpoint=None):
    dest=folder/(name+'.json')
    if dest.exists():return json.loads(dest.read_text())
    start=time.perf_counter();model=None;reprname='pca48'
    if checkpoint:
        ck=torch.load(checkpoint,weights_only=True);model=p.make_svae(pca);model.load_state_dict(ck['state_dict']);model.eval();reprname='svae'
    x=data['features'];te=p.ids(data,'test').numpy();tr=p.ids(data,'train').numpy();testset=set(te.tolist());std=pca['std'].cuda()[None,:,None,None,None]
    features=[];errors=[];episode_ids=[data['manifest'][i]['episode'] for i in te]
    for start_i in range(0,len(x),2):
        z=x[start_i:start_i+2].cuda().float();lat,rec=p.representation(z,reprname,pca,model);features.append(vector(lat))
        mse=((rec-z)/std).square().mean((1,3,4)).cpu().numpy()
        for j in range(len(z)):
            if start_i+j in testset:errors.append(mse[j])
    visuals=np.concatenate(features);state=data['state'].numpy();allvec=np.concatenate([visuals,state],axis=1);probe=fit_probe(allvec,data)
    y=data['target'].numpy()[te];out={'condition':name,'episodes':episode_ids,'alpha':probe['alpha'],'validation_probe_mse':float(probe['val_mse']),'reconstruction':{},'probes':{}}
    errors=np.asarray(errors)
    out['reconstruction']={'all_mse':float(errors.mean()),'condition_mse':float(errors[:,0].mean()),'pooled_target_mse':float(errors[:,1:].mean()),'per_clip':errors.tolist()}
    for variant in ['clean','noise10','brightness06']:
        if variant=='clean':vec=allvec[te]
        else:
            vs=[]
            for chunk in corrupt[variant].split(2):
                lat,_=p.representation(chunk.cuda().float(),reprname,pca,model);vs.append(vector(lat))
            vec=np.concatenate([np.concatenate(vs),state[te]],axis=1)
        pred=predict(probe,vec);error=((pred-y)/probe['ys'])**2;episode_losses=[float(error[np.array(episode_ids)==e].mean()) for e in sorted(set(episode_ids))]
        metrics=p.probe_metrics(pred,y,probe['ys']);metrics.update({'episode_ids':sorted(set(episode_ids)),'episode_losses':episode_losses,'per_clip_losses':error.mean(1).tolist(),'prediction':pred.tolist()})
        out['probes'][variant]=metrics
    # Controls are fit only once per task; recorded on every row for provenance simplicity.
    proprio=fit_probe(state,data);out['proprio_only']=p.probe_metrics(predict(proprio,state[te]),y,proprio['ys'])
    out['wall_seconds']=time.perf_counter()-start;p.write_json(dest,out)
    print('EVAL',name,json.dumps({k:v for k,v in out['reconstruction'].items() if k!='per_clip'}),{k:round(v['normalized_mse'],5) for k,v in out['probes'].items()},flush=True)
    del model;gc.collect();torch.cuda.empty_cache();return out

def checks():
    torch.manual_seed(900)
    out={k:torch.randn(2,c,3,2,2,requires_grad=True) for k,c in [('recon',8),('target',8),('mu',4),('logvar',4)]}
    a,_=weighted_loss(out,.01,1);b,_=p.svae_loss(out,beta=.01)
    assert torch.equal(a,b)
    w,_=weighted_loss(out,.01,4)
    terms=[p.svae_loss({k:v[:,:,i:i+1] for k,v in out.items()},beta=.01)[0] for i in range(3)]
    assert torch.allclose(w,(4*terms[0]+terms[1]+terms[2])/6,atol=1e-6)
    w.backward();assert torch.isfinite(out['recon'].grad).all()
    print('CHECKS: default exact; weighted loss matches explicit frame sum; finite gradient',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');parser.add_argument('--task',choices=TASKS);a=parser.parse_args();checks()
    if a.check_only:return
    p.ensure_gpu();RUNROOT.mkdir(parents=True,exist_ok=True);total=time.perf_counter()
    for task in [a.task] if a.task else TASKS:
        folder=RUNROOT/task;folder.mkdir(exist_ok=True);p.DATA_TASK=task;p.OUT=folder
        if task=='adjust_bottle':
            base=ROOT/'results/pilot-adjust-bottle-v1'
            for filename in ['features.pt','manifest.json','pca.pt','extraction.json']:
                link=folder/filename
                if not link.exists():link.symlink_to(base/filename)
        p.extract();data=p.get_data();pca=p.fit_pca(data);corrupt=corruptions(data,folder)
        results=folder/'evaluation';results.mkdir(exist_ok=True);evaluate(data,pca,corrupt,results,'pca48')
        for seed in SEEDS:
            for weight in [1,4]:
                name=f'svae-w{weight}-s{seed}';ck=train(data,pca,seed,weight,folder/name);evaluate(data,pca,corrupt,results,name,ck)
        del data,pca,corrupt;gc.collect();torch.cuda.empty_cache()
        p.write_json(folder/'complete.json',{'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'seeds':SEEDS,'steps':STEPS})
    p.write_json(RUNROOT/'last-run.json',{'wall_seconds':time.perf_counter()-total,'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
