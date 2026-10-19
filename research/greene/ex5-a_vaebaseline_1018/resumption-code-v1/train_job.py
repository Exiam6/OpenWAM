"""Fixed-endpoint VAE training using the unchanged matched baseline loss/input code."""
import argparse
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import shutil
import signal
import time

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
BASE=ROOT.parent/'rae-factorial-20261010'

def admission(stage):
    from build_cache import sha
    path=ROOT/'gpu-registration.json';assert path.exists(),'GPU registration absent; CPU-only stage'
    from corrected_protocol import load_registration
    p=load_registration();assert p['gpu_launch_permitted']
    assert ('training_smoke' if stage=='smoke' else 'full_training') in p['released_stages']
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
    for name,digest in p['training_files'].items():assert sha(name)==digest,name
    assert str(Path(__file__).resolve()) in p['training_files'] and str(BASE/'common.py') in p['training_files']
    cache=json.loads((ROOT/'vae-cache/result.json').read_text());assert cache['passed'] and cache['completed']==6424
    assert sha(cache['cache'])==cache['sha256']
    assert stage=='full','Original smoke is retained; no smoke retry in corrected full protocol'
    from corrected_protocol import verify_training_release, assert_runtime
    verify_training_release();assert_runtime()
    for path in p['pause_paths']:assert not Path(path).exists()
    return p

def main(seed,stage):
    p=admission(stage)
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from matched_model import build_cpu
    from build_cache import sha
    spec=importlib.util.spec_from_file_location('frozen_baseline_common',BASE/'common.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    run=ROOT/('training-smoke' if stage=='smoke' else 'runs')/f'vae48_native-seed{seed}';run.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();limit=1800 if stage=='smoke' else 10800;result=dict(complete=False,seed=seed,stage=stage,corrected_protocol_sha256=sha(Path(__file__).parent/'protocol.json'),runtime_manifest_sha256=sha(ROOT/'corrected-engine-v1-manifest.json'))
    def guard():
        assert time.monotonic()-started<limit-60
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
        for x in p['pause_paths']:assert not Path(x).exists()
    def stop(*_):raise TimeoutError('Training cap or termination signal')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(limit-60)
    try:
        cache=torch.load(ROOT/'vae-cache/cache.pt',map_location='cpu',weights_only=False,mmap=True)
        # Same implementation and loss as compact baseline, different verified latent tensor.
        inputs=dict(cache,svae=cache['vae']);model,cfg,initialization=build_cpu(seed)
        common.write(run/'initialization.json',initialization)
        common.activate(model)
        params=[x for x in model.parameters() if x.requires_grad]
        opt=torch.optim.AdamW(params,lr=3e-5,betas=(.9,.95),weight_decay=.01)
        batches=np.load(BASE/'runs'/f'raw768_native-seed{seed}'/'batches.npy');assert batches.shape==(20000,2,3)
        np.save(run/'batches.npy',batches)
        steps=3 if stage=='smoke' else 20000
        if stage=='full':common.write(run/'initial-dev.json',common.dev_metrics(model,inputs,'svae48_native'))
        model.train();model.freeze_modules(list(cfg.model.freeze));torch.cuda.reset_peak_memory_stats()
        training_started=time.monotonic();gradients={}
        with (run/'metrics.jsonl').open('x') as log:
            for step in range(1,steps+1):
                guard()
                scale=1. if stage=='smoke' else (step/1000 if step<=1000 else .01+.99*.5*(1+math.cos(math.pi*(step-1000)/19000)))
                for group in opt.param_groups:group['lr']=3e-5*scale
                opt.zero_grad(set_to_none=True);totals=dict(loss=0.,loss_video=0.,loss_action=0.)
                for micro,ids in enumerate(batches[step-1]):
                    batch=common.inputs_for(inputs,ids,'svae48_native')
                    values=common.paired_loss(model,batch,seed*1000000+step*2+micro)
                    (values['loss']/2).backward()
                    for key in totals:totals[key]+=float(values[key])/2
                if stage=='smoke' and step==steps:
                    wanted=['video_backbone.dit.patch_embedding.weight','video_backbone.dit.blocks.0.self_attn.q.weight','action_backbone.action_encoder.weight']
                    named=dict(model.named_parameters())
                    for name in wanted:
                        value=named[name].grad;assert value is not None and torch.isfinite(value).all() and value.float().norm()>0
                        gradients[name]=float(value.float().norm())
                    assert all(x.grad is None for x in model.video_backbone.video_encoder.parameters())
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
                if step==1 or step%25==0 or stage=='smoke':
                    row=dict(seed=seed,step=step,steps=steps,**totals,gradient_norm=float(norm),learning_rate=3e-5*scale,training_seconds=time.monotonic()-training_started,occupied_seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)
                    log.write(json.dumps(row)+'\n');log.flush();common.write(run/'progress.json',row)
        if stage=='full':
            final=common.dev_metrics(model,inputs,'svae48_native');common.write(run/'final-dev.json',final)
        guard()
        for path in (ROOT/'templates/vae48_native').iterdir():
            dest=run/path.name
            if path.is_dir():shutil.copytree(path,dest)
            else:shutil.copy2(path,dest)
        shutil.copy2(BASE/'normalization_stats.npy',run/'normalization_stats.npy')
        checkpoint=run/f'checkpoint_step_{steps}.safetensors'
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
        model.load_checkpoint(str(checkpoint))
        result.update(complete=True,steps=steps,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),strict_reload=True,
                      gradients=gradients,training_seconds=time.monotonic()-training_started,policy_success_claim=False)
    except Exception as error:result['error']=repr(error);raise
    finally:
        signal.alarm(0);result['occupied_seconds']=time.monotonic()-started
        (run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,43,44],required=True);parser.add_argument('--stage',choices=['smoke','full'],required=True);parser.add_argument('--check-admission',action='store_true');a=parser.parse_args()
    if a.check_admission:
        try:admission(a.stage);print('admitted')
        except AssertionError as e:print(str(e));raise SystemExit(2)
    else:main(a.seed,a.stage)
