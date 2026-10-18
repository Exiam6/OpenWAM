"""Training-only tiny-fit gate; this checkpoint is never reused by the full runs."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import signal
import time

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018');BASE=ROOT.parent/'rae-factorial-20261010'

def main():
    from build_cache import sha
    path=ROOT/'gpu-registration.json';assert path.exists(),'CPU-only: no GPU registration'
    p=json.loads(path.read_text());assert p['gpu_launch_permitted'] and 'tiny_fit' in p['released_stages']
    for name,digest in p['tiny_files'].items():assert sha(name)==digest,name
    assert str(Path(__file__).resolve()) in p['tiny_files']
    smoke=ROOT/'training-smoke/vae48_native-seed42';s=json.loads((smoke/'result.json').read_text())
    assert s['complete'] and s['steps']==3 and s['strict_reload']
    assert sha(s['checkpoint'])==s['checkpoint_sha256']
    # The parent launcher owns the combined smoke+tiny1800s cap; this child uses its remaining allowance.
    budget=json.loads((ROOT/'tiny-budget.json').read_text())
    assert p['tiny_total_seconds']==1800 and budget['smoke_result_sha256']==sha(smoke/'result.json')
    assert budget['prior_occupied_seconds']>=s['occupied_seconds']
    allowance=budget['remaining_seconds'];assert 60<allowance<=1800-budget['prior_occupied_seconds']
    import numpy as np
    import torch
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.denoise_schedule import make_schedule
    out=ROOT/'tiny-fit';out.mkdir(exist_ok=False);start=time.monotonic();result=dict(passed=False)
    def guard():
        assert time.monotonic()-start<allowance-60
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
        for x in p['pause_paths']:assert not Path(x).exists()
    def stop(*_):raise TimeoutError('Tiny-fit cap or termination')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(max(1,int(allowance-60)))
    try:
        spec=importlib.util.spec_from_file_location('frozen_baseline_common',BASE/'common.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        cache=torch.load(ROOT/'vae-cache/cache.pt',map_location='cpu',weights_only=False,mmap=True);inputs=dict(cache,svae=cache['vae'])
        tasks=common.P['tasks'];ids=[next(i for i,m in enumerate(cache['metadata']) if m['task']==task and m['split']=='train') for task in tasks]
        cfg,model=load_from_checkpoint_dir(str(smoke),device='cuda:0',ckpt_name=Path(s['checkpoint']).name)
        assert not model.binary_command_dims
        common.activate(model)
        prompt_cache={}
        def measure():
            model.eval();rows=[]
            for i in ids:
                guard();seed=9210180+i
                schedule=make_schedule('sync',video_scheduler=model.video_scheduler,action_scheduler=model.action_scheduler,num_steps=10,shift=5.,shift_video=5.)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    action=model.generate(schedule,common.P['prompts'][cache['metadata'][i]['task']],first_frame_latents=cache['vae'][i:i+1,:,:1].to('cuda',torch.bfloat16),
                        fuse_vae_embedding_in_latents=True,num_clean_prefix_frames=0,first_frame_image=None,num_frames=9,action_num_frames=33,height=384,width=320,
                        seed=seed,tiled=False,num_inference_steps=10,shift=5.,decode_video=False,prompt_embed_cache=prompt_cache,
                        proprio=cache['proprio'][i:i+1].to('cuda',torch.bfloat16))['actions']
                action=np.asarray(action);assert action.shape==(32,20) and np.isfinite(action).all()
                normalized=np.asarray(model.normalizer.normalize(action));truth=cache['actions'][i].numpy()
                rows.append(dict(cache_index=i,task=cache['metadata'][i]['task'],generated_action_mse=float(np.mean((normalized-truth)**2))))
            # Inference schedules must not leak into the next native training loss.
            model.init_training_schedulers();model.train();model.freeze_modules(list(cfg.model.freeze))
            loss=common.paired_loss(model,common.inputs_for(inputs,ids,'svae48_native'),921018)
            return dict(rows=rows,flow_action_loss=float(loss['loss_action']),mean_generated_mse=float(np.mean([x['generated_action_mse'] for x in rows])))
        before=measure();params=[x for x in model.parameters() if x.requires_grad]
        opt=torch.optim.AdamW(params,lr=3e-5,betas=(.9,.95),weight_decay=.01)
        with (out/'metrics.jsonl').open('x') as log:
            for step in range(1,513):
                guard();opt.zero_grad(set_to_none=True)
                for micro in range(2):
                    loss=common.paired_loss(model,common.inputs_for(inputs,ids,'svae48_native'),42000000+step*2+micro);(loss['loss']/2).backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
                if step%32==0:log.write(json.dumps(dict(step=step,loss_action=float(loss['loss_action']),gradient_norm=float(norm),seconds=time.monotonic()-start))+'\n');log.flush()
        after=measure()
        passed=after['flow_action_loss']<=.25*before['flow_action_loss'] and after['mean_generated_mse']<=.25*before['mean_generated_mse'] and all(b['generated_action_mse']<=.5*a['generated_action_mse'] for a,b in zip(before['rows'],after['rows']))
        result.update(passed=passed,before=before,after=after,steps=512,training_indices=ids,checkpoint_reused_for_full_training=False,
            criterion='Fixed512steps on first train window/task; >=4x lower fixed-noise action flow loss and mean generated normalized-action MSE; >=2x lower generated MSE for each task. No closed-loop success claim.')
    except Exception as error:result['error']=repr(error);raise
    finally:
        signal.alarm(0);result['occupied_seconds']=time.monotonic()-start;(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
