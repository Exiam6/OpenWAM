"""Single-GPU, bounded training-only BF16 contract and throughput preflight."""
import datetime
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from objective_v2 import waypoint_objective, with_waypoint_objective, spatial_pool
import inspect
import objective as original_objective

ROOT = Path(os.environ['WAM_ROOT'])
L = ROOT / 'results/layers-20260922'
OUT = ROOT / 'results/goalaux-20260922/gpu-preflight-v2'
TASKS = ['adjust_bottle', 'handover_block', 'place_object_basket']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')


def main():
    p=json.loads((OUT/'protocol.json').read_text())
    assert not (OUT/'started.json').exists(),'Do not replay preflight'
    assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(p['deadline'])
    assert torch.cuda.device_count()==1
    for relative,expected in p['source_sha256'].items():assert sha(ROOT/relative)==expected,relative
    write('started.json',{'started_at':datetime.datetime.now().astimezone().isoformat(),'protocol_sha256':sha(OUT/'protocol.json')})
    start=time.monotonic();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    native_path=ROOT/'OpenWAM/openwam/model/video_backbone/encoder/svae/model.py'
    spec=importlib.util.spec_from_file_location('native_svae',native_path)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    ckpt=torch.load(L/'L12/seed42/final.pt',map_location='cpu',weights_only=False)
    labels=json.loads((L/'initial-goal-training-labels.json').read_text())
    assert labels['dev_or_confirmation_read'] is False
    target=[];clips=[];records=[];data_hashes={}
    for task in TASKS:
        cache=L/'features'/task/'episode3.pt'
        data=torch.load(cache,map_location='cpu',weights_only=False)
        assert all(r['task']==task and r['episode']==3 and r['split']=='train' for r in data['manifest'])
        assert data['manifest'][0]['frame']==0
        assert tuple(data['features']['L12'].shape)==(12,768,3,24,20)
        clips.append(data['features']['L12'].float())
        rows=[r for r in labels['records'] if r['task']==task]
        assert len(rows)==35 and all(r['valid'] for r in rows)
        yy=np.array([r['native_endpose_position'][:2] for r in rows],dtype=np.float64)
        y=next(r for r in rows if r['episode']==3)['native_endpose_position'][:2]
        target.append((np.asarray(y)-yy.mean(0))/yy.std(0).clip(1e-6))
        records.append({'task':task,'episode':3,'frame':0,'split':'train'})
        data_hashes[str(cache.relative_to(ROOT))]=sha(cache)
    clips=torch.stack(clips).cuda();target=torch.tensor(np.stack(target),dtype=torch.float32,device='cuda');task_idx=torch.arange(3,device='cuda')
    initial=clips[:,0,:,:1].contiguous()
    def setup():
        torch.manual_seed(42)
        model=native.SVAE(**ckpt['model_config']).cuda();model.load_state_dict(ckpt['state_dict'],strict=True);model.train()
        heads=torch.nn.ModuleList([torch.nn.Linear(192,2) for _ in TASKS]).cuda()
        opt=torch.optim.AdamW(list(model.parameters())+list(heads.parameters()),lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
        return model,heads,opt
    def step(model,heads,opt,index,weight):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(clips[:,index])
            native_loss,_=native.svae_loss(output,beta=1e-4,cos_weight=1.)
            if weight:
                mean=model.encode_mean(initial)
                loss=with_waypoint_objective(native_loss,mean,heads,task_idx,target,weight)
            else:
                loss=with_waypoint_objective(native_loss,None,None,None,None,0.)
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(heads.parameters()),1.)
        assert torch.isfinite(norm)
        opt.step()
        return float(loss.detach())
    previous=ROOT/'results/goalaux-20260922/gpu-preflight'
    prior_log=(previous/'run.log').read_text()
    assert 'BF16_ZERO_WEIGHT_EXACT' in prior_log
    assert 'adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation' in prior_log
    assert (previous/'exit-code.txt').read_text().strip()=='1'
    assert inspect.getsource(with_waypoint_objective)==inspect.getsource(original_objective.with_waypoint_objective)
    parity={'exact':True,'parameter_gradients':80,'cpu_and_cuda_rng_equal':True,'one_adamw_update_equal':True,
            'inherited_from':'gpu-preflight/run.log','rerun':False,'reason':'Zero-weight wrapper source identical; only positive-weight pooling implementation changed'}
    # Match the original pool on existing TRAIN-only tensors, including CPU grads.
    torch.manual_seed(123)
    cpu=clips[:,0,:,:1].cpu()[:, :48, 0].requires_grad_(True)
    ref=torch.nn.functional.adaptive_avg_pool2d(cpu,(2,2))
    got=spatial_pool(cpu)
    assert torch.allclose(got,ref,rtol=1e-6,atol=1e-6)
    grad_ref=torch.autograd.grad(ref.square().sum(),cpu,retain_graph=True)[0]
    grad_got=torch.autograd.grad(got.square().sum(),cpu)[0]
    assert torch.allclose(grad_ref,grad_got,rtol=1e-6,atol=1e-7)
    cpu_forward_max=float((got-ref).abs().max());cpu_grad_max=float((grad_ref-grad_got).abs().max())
    del cpu,ref,got,grad_ref,grad_got
    model,heads,opt=setup()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        mean=model.encode_mean(initial)
        ref_pool=torch.nn.functional.adaptive_avg_pool2d(mean.detach()[:,:,0],(2,2))
        got_pool=spatial_pool(mean.detach()[:,:,0])
        assert torch.allclose(got_pool,ref_pool,rtol=2*torch.finfo(torch.bfloat16).eps,atol=torch.finfo(torch.bfloat16).eps/8)
        gpu_pool_max=float((got_pool-ref_pool).abs().max())
        aux=waypoint_objective(mean,heads,task_idx,target)
    aux.backward()
    enc=[q.grad for n,q in model.named_parameters() if n.startswith('enc_')]
    assert all(g is not None and torch.isfinite(g).all() for g in enc)
    enc_norm=float(torch.sqrt(sum(g.float().square().sum() for g in enc)));assert enc_norm>0
    assert all(q.grad is None for n,q in model.named_parameters() if n.startswith('dec_'))
    heads_norm=float(torch.sqrt(sum(q.grad.float().square().sum() for q in heads.parameters())));assert heads_norm>0
    del model,heads,opt,aux,enc,mean,ref_pool,got_pool;gc.collect();torch.cuda.empty_cache()
    profile={}
    for label,weight in [('reconstruction_only',0.),('goal_auxiliary',0.1)]:
        model,heads,opt=setup();torch.manual_seed(20260922)
        torch.cuda.reset_peak_memory_stats();elapsed=[];last=None
        for index in range(40):
            torch.cuda.synchronize();tick=time.monotonic()
            last=step(model,heads,opt,index%12,weight)
            torch.cuda.synchronize();dt=time.monotonic()-tick
            if index>=10:elapsed.append(dt)
        med=float(np.median(elapsed));profile[label]={'weight':weight,'warmup_updates':10,'timed_updates':30,'timed_seconds':elapsed,
            'median_seconds_per_update':med,'p95_seconds_per_update':float(np.quantile(elapsed,.95)),
            'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,'last_loss_diagnostic_only':last,
            'rough_2000_update_seconds':2000*med}
        print('PROFILE',label,json.dumps(profile[label]),flush=True)
        del model,heads,opt;gc.collect();torch.cuda.empty_cache()
    assert sha(L/'L12/seed42/final.pt')==p['source_sha256']['results/layers-20260922/L12/seed42/final.pt']
    result={'passed':True,'finished_at':datetime.datetime.now().astimezone().isoformat(),'elapsed_seconds':time.monotonic()-start,
        'dtype':'CUDA autocast BF16; FP32 parameters','deterministic_algorithms':True,'gpu_name':torch.cuda.get_device_name(0),
        'torch_version':torch.__version__,'device_count':1,'cpu_threads':4,'peak_cpu_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'batch_shape':[3,768,3,24,20],'auxiliary_batch_shape':[3,768,1,24,20],'records':records,'cache_sha256':data_hashes,
        'zero_weight_parity':parity,'pooling_equivalence':{'cpu_fp32_forward_max_abs':cpu_forward_max,'cpu_fp32_gradient_max_abs':cpu_grad_max,'gpu_bf16_forward_max_abs':gpu_pool_max,'cpu_rtol':1e-6,'cpu_atol':1e-6,'cpu_gradient_atol':1e-7,'gpu_rtol':2*torch.finfo(torch.bfloat16).eps,'gpu_atol':torch.finfo(torch.bfloat16).eps/8},'auxiliary_encoder_gradient_norm':enc_norm,'auxiliary_head_gradient_norm':heads_norm,
        'profiles':profile,'checkpoint_unchanged':True,'new_checkpoint_saved':False,'dev_or_confirmation_scored':False,
        'limits':['Timing uses three existing training episodes, resident feature cache, oneGPU; excludes extraction/IO/collection/readout',
                  'Temporary profiling updates are discarded; no fitted representation or policy gain claim',
                  'One checkpoint seed; no all-seed or distributed parity claim']}
    write('result.json',result);print('PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':main()
