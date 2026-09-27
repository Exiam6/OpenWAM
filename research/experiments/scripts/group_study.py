#!/usr/bin/env python3
"""Preregistered grouped-loss follow-up; existing development tests remain closed."""
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import control_study as c
n=c.n
p=c.p
OUT=p.ROOT/'results/groups-20260921'
FRESH='pick_dual_bottles'
STEPS=c.STEPS
train_stats=c.train_stats
make_head=c.make_head
combined_loss=c.combined_loss
gpu_guard=c.gpu_guard


def group_auxiliary_loss(head, mu, state, target, stats, rho):
    if rho == .25:
        return c.auxiliary_loss(head, mu, state, target, stats)
    state=(state-stats['state']['mean'])/stats['state']['std']
    target=(target-stats['target']['mean'])/stats['target']['std']
    pred=head(torch.cat([c.current_vector(mu),state],1)).float()
    errors=(pred-target.float()).square()
    return (1-rho)*errors[:,:6].mean()+rho*errors[:,6:].mean()


def train(data, pca, weight, seed, folder, rho):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'final.pt'
    if path.exists():
        return path
    gpu_guard()
    torch.manual_seed(seed)
    model = p.make_svae(pca)
    head = make_head(data['state'].shape[1], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .99), weight_decay=1e-4)
    hopt = torch.optim.AdamW(head.parameters(), lr=1e-4, betas=(.9, .99), weight_decay=1e-4)
    stats = train_stats(data)
    stats_gpu = {key: {k: v.cuda() for k, v in row.items()} for key, row in stats.items()}
    rng = torch.Generator().manual_seed(seed)
    train_ids, val_ids = p.ids(data, 'train'), p.ids(data, 'val')
    x = data['features']; initial = n.reconstruction(model, x, val_ids)
    start = time.perf_counter(); history = []; torch.cuda.reset_peak_memory_stats()
    for step in range(1, STEPS + 1):
        selected = train_ids[torch.randint(len(train_ids), (2,), generator=rng)]
        z = x[selected].cuda().float(); state = data['state'][selected].cuda(); target = data['target'][selected].cuda()
        model.train(); head.train(); opt.zero_grad(set_to_none=True); hopt.zero_grad(set_to_none=True)
        lr = 1e-4 * min(1., step / 20) * (.95 + .05 * np.cos(np.pi * step / STEPS))
        for optimizer in [opt, hopt]:
            for group in optimizer.param_groups: group['lr'] = lr
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(z)
            base, _ = p.svae_loss(out, beta=1e-4 * min(1., step / (STEPS * .2)), cos_weight=1.)
            head_mu = out['mu'].detach() if weight == 0 else out['mu']
            aux = group_auxiliary_loss(head, head_mu, state, target, stats_gpu, rho)
            loss = combined_loss(base, aux, weight)
        assert torch.isfinite(loss), (step, base, aux)
        loss.backward()
        if weight == 0:
            aux.backward()  # Train the reference head without changing compressor gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        opt.step(); hopt.step()
        if step % 200 == 0:
            gpu_guard()
            row = {'step': step, 'base_loss': float(base.detach()), 'auxiliary_loss': float(aux.detach()),
                   'val_reconstruction': n.reconstruction(model, x, val_ids), 'elapsed_seconds': time.perf_counter() - start}
            history.append(row); p.write_json(folder / 'curve.json', history)
            print('TRAIN', folder.parent.name, folder.name, json.dumps(row), flush=True)
    torch.save({'format_version': 2, 'model_config': model.config_dict(),
                'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'step': STEPS, 'seed': seed, 'auxiliary_weight': weight}, path)
    torch.save({'head_state_dict': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, 'stats': stats}, folder / 'auxiliary.pt')
    row = {'seed': seed, 'auxiliary_weight': weight, 'steps': STEPS, 'initial_validation': initial,
           'wall_seconds': time.perf_counter() - start, 'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
           'head_parameters': sum(v.numel() for v in head.parameters()), 'head_hidden': 128,
           'input_state_dim': data['state'].shape[1], 'target_dim': data['target'].shape[1], 'gripper_share': rho}
    if weight == 0 and (n.RUNROOT / folder.parent.name / f'svae-w1-s{seed}' / 'final.pt').exists():
        old = torch.load(n.RUNROOT / folder.parent.name / f'svae-w1-s{seed}' / 'final.pt', weights_only=True)
        row['previous_baseline_weight_max_abs'] = max(float((model.state_dict()[k].cpu() - v).abs().max()) for k, v in old['state_dict'].items())
    p.write_json(folder / 'training.json', row)
    del model, head, opt, hopt, out; gc.collect(); torch.cuda.empty_cache()
    return path



def probe_groups(data, vec, split):
    x=np.concatenate([vec,data['state'].numpy()],1)
    out={}; ii=p.ids(data,split).numpy()
    episodes=np.array([data['manifest'][i]['episode'] for i in ii])
    for name,cols,scale in [('translation',slice(0,6),1000.),('gripper',slice(6,8),1.)]:
        blind=dict(data);blind['target']=data['target'][:,cols].clone()
        blind['target'][p.ids(data,'test')]=float('nan')
        probe=n.fit_probe(x,blind)
        target=data['target'][:,cols].numpy()[ii]
        pred=n.predict(probe,x[ii]);err=pred-target;norm=(err/probe['ys'])**2
        row={'alpha':float(probe['alpha']),'validation_mse':float(probe['val_mse']),
             'normalized_mse':float(norm.mean()),'rmse':float(scale*np.sqrt((err**2).mean()))}
        if split=='test':
            unique=np.unique(episodes)
            transition=np.max(np.abs(data['target'][ii,6:].numpy()-data['state'][ii,-2:].numpy()),1)>=.05
            row.update({'episode_ids':unique.tolist(),
                        'episode_losses':[float(norm[episodes==e].mean()) for e in unique],
                        'episode_raw_mse':[float((err[episodes==e]**2).mean()) for e in unique],
                        'transition_clips':int(transition.sum()),'test_clips':len(ii),
                        'transition_normalized_mse':float(norm[transition].mean()) if transition.any() else None,
                        'transition_rmse':float(scale*np.sqrt((err[transition]**2).mean())) if transition.any() else None})
        out[name]=row
    return out


def validation(data,pca,folder,ck=None):
    path=folder/'group-validation.json'
    if path.exists():return json.loads(path.read_text())
    if not (folder/'vectors.npy').exists():
        np.save(folder/'vectors.npy',c.vectors(data,pca,'svae',ck))
    result=probe_groups(data,np.load(folder/'vectors.npy'),'val')
    p.write_json(path,result)
    print('GROUP VALIDATION',folder.parent.name,folder.name,json.dumps(result),flush=True)
    return result


def select(vals):
    scores={}
    for rho in [.5,.75]:
        row={}
        for group in ['translation','gripper']:
            for ref in ['aux0','aux1']:
                ratios={t:float(np.mean([vals[t][f'g{rho}-s{s}'][group]['normalized_mse'] for s in n.SEEDS])/np.mean([vals[t][f'{ref}-s{s}'][group]['normalized_mse'] for s in n.SEEDS])) for t in n.TASKS}
                row[f'{group}_ratios_to_{ref}']=ratios
                row[f'{group}_macro_to_{ref}']=float(np.mean(list(ratios.values())))
        row['translation_eligible']=row['translation_macro_to_aux1']<=1.10 and max(row['translation_ratios_to_aux1'].values())<=1.20
        row['promotion_gate']=row['translation_eligible'] and max(row['gripper_ratios_to_aux0'].values())<=1.05 and row['translation_macro_to_aux0']<=.95
        scores[str(rho)]=row
    eligible=[r for r in [.5,.75] if scores[str(r)]['translation_eligible']]
    if eligible:chosen=min(eligible,key=lambda r:scores[str(r)]['gripper_macro_to_aux1'])
    else:chosen=min([.5,.75],key=lambda r:max(scores[str(r)]['translation_macro_to_aux1'],scores[str(r)]['gripper_macro_to_aux1']))
    return {'selected_rho':chosen,'eligible_selection':bool(eligible),'promotion_gate':scores[str(chosen)]['promotion_gate'],
            'scores':scores,'selected_before_fresh_training_and_test':True,'selected_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}


def group_checks(data,pca):
    c.checks(data,pca,OUT)
    torch.manual_seed(42);model=p.make_svae(pca);head=make_head(data['state'].shape[1],42)
    stats={key:{k:v.cuda() for k,v in row.items()} for key,row in train_stats(data).items()}
    x=data['features'][:2].cuda().float();state=data['state'][:2].cuda();target=data['target'][:2].cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        mu=model.encode_mean(x)
        original=c.auxiliary_loss(head,mu,state,target,stats)
        assert torch.equal(original,group_auxiliary_loss(head,mu,state,target,stats,.25))
        t=group_auxiliary_loss(head,mu,state,target,stats,0.)
        g=group_auxiliary_loss(head,mu,state,target,stats,1.)
        for rho in [.5,.75]:
            actual=group_auxiliary_loss(head,mu,state,target,stats,rho)
            assert torch.allclose(actual,(1-rho)*t+rho*g,atol=1e-6,rtol=1e-6)
    del model,head,x,mu;gc.collect();torch.cuda.empty_cache()
    p.write_json(OUT/'group-checks.json',{'original_loss_branch_exact':True,'group_weight_formula_checked':True})


def summarize_fresh(rows,chosen):
    result={'selected_rho':chosen,'methods':{},'paired_comparisons':{}}
    for name in ['aux0','aux1',f'g{chosen}']:
        result['methods'][name]={g:{k:(float(np.mean([rows[f'{name}-s{s}'][g][k] for s in n.SEEDS])) if rows[f'{name}-s42'][g][k] is not None else None) for k in ['normalized_mse','rmse','transition_normalized_mse','transition_rmse']} for g in ['translation','gripper']}
    # Paired episode bootstrap, seeds fixed. All methods use the same 10 episodes.
    draw=np.random.default_rng(20260921).integers(0,10,size=(10000,10))
    for a,b in [('aux1','aux0'),(f'g{chosen}','aux0'),(f'g{chosen}','aux1')]:
        comparison={}
        for group in ['translation','gripper']:
            ra=np.mean([rows[f'{a}-s{s}'][group]['episode_losses'] for s in n.SEEDS],axis=0)
            rb=np.mean([rows[f'{b}-s{s}'][group]['episode_losses'] for s in n.SEEDS],axis=0)
            assert len(ra)==len(rb)==10
            assert all(rows[f'{a}-s{s}'][group]['episode_ids']==rows[f'{b}-s{s}'][group]['episode_ids'] for s in n.SEEDS)
            delta=ra-rb;means=delta[draw].mean(1)
            am=result['methods'][a][group]['rmse'];bm=result['methods'][b][group]['rmse']
            comparison[group]={'rmse_change_percent':100*(am/bm-1),'normalized_mse_delta':float(delta.mean()),
                               'paired_episode_95ci_delta':np.quantile(means,[.025,.975]).tolist()}
        result['paired_comparisons'][a+'_vs_'+b]=comparison
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);p.ensure_gpu();gpu_guard();start=time.perf_counter()
    p.write_json(OUT/'protocol.json',{'sha256':{f:hashlib.sha256((p.ROOT/f).read_bytes()).hexdigest() for f in ['GROUP_PROTOCOL.md','scripts/group_study.py','scripts/control_study.py','scripts/pilot.py','scripts/night.py']},'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'new_training_run_cap':28,'fresh_task':FRESH})
    data,pca=c.load_task(n.TASKS[0]);group_checks(data,pca)
    audit=OUT/'numeric-audit'/n.TASKS[0]/'aux1-s42'
    ck=train(data,pca,1.,42,audit,.25)
    before=torch.load(c.OUT/n.TASKS[0]/'aux1-s42/final.pt',weights_only=True)['state_dict']
    after=torch.load(ck,weights_only=True)['state_dict']
    difference=max(float((before[k]-after[k]).abs().max()) for k in before)
    p.write_json(OUT/'reuse-audit.json',{'original_checkpoint_max_abs':difference,'passed':difference==0.})
    assert difference==0.,('Old controls cannot be reused on this device',difference)
    del data,pca,before,after;gc.collect();torch.cuda.empty_cache()
    vals={}
    for task in n.TASKS:
        data,pca=c.load_task(task);vals[task]={}
        for seed in n.SEEDS:
            for name in ['aux0','aux1']:
                folder=c.OUT/task/f'{name}-s{seed}'
                vals[task][f'{name}-s{seed}']=probe_groups(data,np.load(folder/'vectors.npy'),'val')
            for rho in [.5,.75]:
                name=f'g{rho}-s{seed}';folder=OUT/task/name
                ck=train(data,pca,1.,seed,folder,rho)
                vals[task][name]=validation(data,pca,folder,ck)
        p.write_json(OUT/'development-validation.json',vals)
        del data,pca;gc.collect();torch.cuda.empty_cache()
    selection=select(vals);p.write_json(OUT/'selection.json',selection)
    print('SELECTION',json.dumps(selection),flush=True)
    chosen=selection['selected_rho']
    # This stage begins only after immutable selection exists.
    fresh=OUT/FRESH;fresh.mkdir(exist_ok=True);p.DATA_TASK=FRESH;p.OUT=fresh
    gpu_guard();p.extract();data=p.get_data();pca=p.fit_pca(data)
    assert data['features'].shape==(600,768,3,24,20)
    for seed in n.SEEDS:
        for name,weight,rho in [('aux0',0.,.25),('aux1',1.,.25),(f'g{chosen}',1.,chosen)]:
            folder=fresh/f'{name}-s{seed}';ck=train(data,pca,weight,seed,folder,rho)
            validation(data,pca,folder,ck)
    p.write_json(OUT/'test-unlock.json',{'selection_sha256':hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),'unlocked_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'all_fresh_training_completed':True})
    rows={}
    for seed in n.SEEDS:
        for name in ['aux0','aux1',f'g{chosen}']:
            folder=fresh/f'{name}-s{seed}'
            row=probe_groups(data,np.load(folder/'vectors.npy'),'test')
            model=p.make_svae(pca);model.load_state_dict(torch.load(folder/'final.pt',weights_only=True)['state_dict'])
            row['reconstruction']=n.reconstruction(model,data['features'],p.ids(data,'test'))
            del model;gc.collect();torch.cuda.empty_cache()
            rows[folder.name]=row;p.write_json(folder/'group-test.json',row)
    summary=summarize_fresh(rows,chosen);summary['selection']=selection;summary['raw_results']=rows
    summary['limits']=['Independent task with per-task fitting, not zero-shot transfer','10 held-out episodes; CIs conditional on 3 fixed training seeds','Achieved-state readout, not policy success; no contact labels','No further group-weight search after this evaluation']
    p.write_json(OUT/'summary.json',summary)
    p.write_json(OUT/'complete.json',{'wall_seconds':time.perf_counter()-start,'training_runs':28,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('COMPLETE',json.dumps(summary['paired_comparisons']),flush=True)


if __name__=='__main__':main()
