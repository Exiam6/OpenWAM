"""Matched full native-policy optimization; fixed final checkpoints and dev loss."""
import argparse
import gc
import math
import shutil
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from common import ROOT, P, activate, build, dev_metrics, guard, inputs_for, paired_loss, sha, stamp, write

parser = argparse.ArgumentParser()
parser.add_argument('variant', choices=P['variants'])
parser.add_argument('seed', type=int)
parser.add_argument('--preflight', action='store_true')
a = parser.parse_args()
run = ROOT / ('preflight' if a.preflight else 'runs') / f'{a.variant}-seed{a.seed}'
run.mkdir(parents=True, exist_ok=True)
assert not (run / 'result.json').exists(), 'Completed run must not be repeated'
guard()
started = time.monotonic()
cache = torch.load(ROOT / 'cache.pt', map_location='cpu', weights_only=False)
initdir = ROOT / 'initialization'
initdir.mkdir(exist_ok=True)
initpath = initdir / f'common-seed{a.seed}.safetensors'
if not initpath.exists():
    base, cfg = build('raw768_native', a.seed, frozen=False)
    common = {n: p.detach().cpu().contiguous() for n, p in base.named_parameters() if p.requires_grad}
    save_file(common, str(initpath))
    del base, common
    gc.collect()
model, cfg = build(a.variant, a.seed)
shared = load_file(str(initpath))
state = model.state_dict()
matched = {k: v for k, v in shared.items() if k in state and v.shape == state[k].shape}
missing, unexpected = model.load_state_dict(matched, strict=False)
assert not unexpected
assert all(torch.equal(model.state_dict()[k], v) for k, v in matched.items())
excluded = sorted(set(shared) - set(matched))
assert all(k.startswith(('video_backbone.dit.patch_embedding.', 'video_backbone.dit.head.head.')) for k in excluded), excluded
write(run / 'initialization.json', dict(variant=a.variant, seed=a.seed, common_sha256=sha(initpath),
      common_exact_tensors=len(matched), dimension_specific_exclusions=excluded,
      published_policy_weights_loaded=False, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
      total_parameters=sum(p.numel() for p in model.parameters())))
del shared, matched, state
activate(model)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=P['training']['learning_rate'], betas=P['training']['betas'], weight_decay=.01)
# Build the complete paired batch schedule before reading any development metric.
rng = np.random.default_rng(a.seed)
by_task = {task: np.array([i for i,m in enumerate(cache['metadata']) if m['split']=='train' and m['task']==task]) for task in P['tasks']}
batches = np.array([[[rng.choice(by_task[t]) for t in P['tasks']] for micro in range(P['training']['accumulation'])] for step in range(P['steps'])])
np.save(run / 'batches.npy', batches)
steps = 3 if a.preflight else P['steps']
warmup = P['training']['warmup_steps']
start_step = 0
resume = run / 'resume.pt'
if resume.exists() and not a.preflight:
    saved = torch.load(resume, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['policy'], strict=False)
    opt.load_state_dict(saved['optimizer'])
    start_step = saved['step']
    del saved
    print('resumed from step', start_step, flush=True)
if not a.preflight and not (run / 'initial-dev.json').exists():
    write(run / 'initial-dev.json', dev_metrics(model, cache, a.variant))
model.train()
model.freeze_modules(list(cfg.model.freeze))
torch.cuda.reset_peak_memory_stats()
train_started = time.monotonic()
last_grad = {}
with (run / 'metrics.jsonl').open('a') as log:
    for step in range(start_step + 1, steps + 1):
        guard()
        if a.preflight:
            scale = 1.
        elif step <= warmup:
            scale = step / warmup
        else:
            scale = .01 + .99 * .5 * (1 + math.cos(math.pi * (step-warmup)/(P['steps']-warmup)))
        for group in opt.param_groups:
            group['lr'] = P['training']['learning_rate'] * scale
        opt.zero_grad(set_to_none=True)
        totals = dict(loss=0., loss_video=0., loss_action=0.)
        micros = len(batches[step-1])
        for micro, ids in enumerate(batches[step-1]):
            out = paired_loss(model, inputs_for(cache, ids, a.variant), a.seed*1000000+step*2+micro)
            (out['loss']/micros).backward()
            for key in totals:
                totals[key] += float(out[key])/micros
        if a.preflight and step == steps:
            wanted = {'trunk_input':'video_backbone.dit.patch_embedding.weight',
                      'trunk_attention':'video_backbone.dit.blocks.0.self_attn.q.weight',
                      'action_input':'action_backbone.action_embedding.weight'}
            for name,param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name.startswith('action_backbone.') and 'action' not in last_grad:
                        value = float(param.grad.float().norm())
                        if value > 0:
                            last_grad['action'] = dict(name=name,norm=value)
                    if name in wanted.values() or 'rae_head.raw_projection.weight' in name or 'rae_head.output.weight' in name:
                        last_grad[name] = float(param.grad.float().norm())
            assert last_grad.get('action'), 'No action gradient'
            assert last_grad['video_backbone.dit.patch_embedding.weight'] > 0
            if a.variant == 'raw768_wide':
                assert last_grad['video_backbone.dit.rae_head.raw_projection.weight'] > 0
                assert all(p.grad is None for p in model.video_backbone.dit.head.parameters())
        grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 25 == 0 or a.preflight:
            row = dict(variant=a.variant, seed=a.seed, step=step, steps=steps, **totals,
                       gradient_norm=float(grad), lr=opt.param_groups[0]['lr'],
                       seconds=time.monotonic()-train_started, peak_gib=torch.cuda.max_memory_allocated()/2**30,updated=stamp())
            log.write(__import__('json').dumps(row)+'\n'); log.flush()
            write(run/'status.json', row); write(ROOT/f"status-gpu{__import__('os').environ.get('STUDY_GPU','')}.json",dict(row,phase='preflight' if a.preflight else 'training'))
            print(__import__('json').dumps(row), flush=True)
        if not a.preflight and step % P['training']['resume_every'] == 0 and step != steps:
            # Atomic replacement; one owned recovery state per run, never a selected checkpoint.
            tmp = resume.with_suffix('.tmp')
            torch.save(dict(step=step,policy={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                            optimizer=opt.state_dict()),tmp)
            tmp.replace(resume)
if a.preflight:
    # Serialization equality for the actual trained policy, including its new head.
    policy = {n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
    save_file(policy,str(run/'policy.safetensors'))
    reloaded = load_file(str(run/'policy.safetensors'))
    assert set(policy)==set(reloaded) and all(torch.equal(policy[n],reloaded[n]) for n in policy)
    write(run/'result.json',dict(complete=True,variant=a.variant,seed=a.seed,steps=steps,gradients=last_grad,
          serialization_exact=True,peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started))
else:
    final = dev_metrics(model,cache,a.variant)
    write(run/'final-dev.json',final)
    # Ordinary self-contained native checkpoint: shared pretrained frozen tensors included.
    for path in (ROOT/'templates'/a.variant).iterdir():
        target=run/path.name
        if path.is_dir():
            shutil.copytree(path,target,dirs_exist_ok=True)
        else:
            shutil.copy2(path,target)
    shutil.copy2(ROOT/'normalization_stats.npy',run/'normalization_stats.npy')
    checkpoint=run/f'checkpoint_step_{steps}.safetensors'
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
    # Full native strict loading catches omitted frozen, action, or head tensors.
    model.load_checkpoint(str(checkpoint))
    write(run/'result.json',dict(complete=True,variant=a.variant,seed=a.seed,steps=steps,
          checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),strict_reload=True,
          final_dev_action_loss=final['task_balanced_action_loss'],final_dev_video_loss=float(np.mean([v['loss_video'] for v in final['per_task'].values()])),
          resumed_from_step=start_step,peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started,
          training_seconds=time.monotonic()-train_started,policy_success_claim=False))
    resume.unlink(missing_ok=True)
print('COMPLETE',a.variant,a.seed,'preflight' if a.preflight else 'training',flush=True)
