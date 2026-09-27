#!/usr/bin/env python3
"""All-task paired summary, using only numerically matched corruption results."""
import json,math,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/night-20260921';OUT=ROOT/'public-results/night-20260921';OUT.mkdir(parents=True,exist_ok=True)
TASKS=['adjust_bottle','handover_block','place_object_basket'];SEEDS=[42,43,44]
torch.set_num_threads(4)
def read(p):return json.loads(p.read_text())
def stats(x):return {'mean':float(np.mean(x)),'std_across_training_seeds':float(np.std(x,ddof=1)) if len(x)>1 else None}
def paired_ci(base,candidate):
    # Average training seeds, THEN cluster windows by episode. No pseudo-replication.
    a=np.asarray(base).mean(0);b=np.asarray(candidate).mean(0);d=b-a;rng=np.random.default_rng(931)
    boot=rng.choice(d,(10000,len(d)),replace=True).mean(1)
    return {'candidate_minus_baseline':float(d.mean()),'episode_paired_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'relative_change_percent':float(100*(b.mean()/a.mean()-1)),'unit':'episode; averaged over 3 paired training seeds','n_episodes':len(d)}
summary={'protocol':'NIGHT_PROTOCOL.md','date':'2026-09-21','source_commit':'7c5861e45cfe1339a0323f0e0b03a3316c37971c','representation':'DINOv3 + S-VAE Study branch; NOT OpenWAM-alpha','steps':2000,'seeds':SEEDS,'tasks':{},'limitations':['exploratory small-budget offline experiment','3 clean simulation tasks, 10 held-out episodes per task','12 complete clips per episode rather than exhaustive windows; temporal sampling matches released config: 33 raw steps, stride 4, 9 video frames','test probes predict achieved state changes, not policy action commands or closed-loop success','fixed 2000-step budget, not convergence-matched','synthetic image perturbations, not LIBERO-Plus','bootstrap intervals condition on trained seeds; seed spread reported separately','initial single-frame corruption outputs excluded; matched encoder batch geometry checked on every held-out clip']}
resources={'training_seconds':0.,'evaluation_seconds':0.,'peak_allocated_gb':0.,'trained_models':0}
for task in TASKS:
    folder=RUN/task;check=read(folder/'corruption-matched-batch-check.json');assert check['all_test_clean_max_abs']<1e-6
    rows={p.stem:read(p) for p in (folder/'evaluation-matched-batch').glob('*.json')};assert len(rows)==7
    data=torch.load(folder/'features.pt',weights_only=False);pca=torch.load(folder/'pca.pt',weights_only=True)
    ids=[i for i,m in enumerate(data['manifest']) if m['split']=='test'];episodes=np.array([data['manifest'][i]['episode'] for i in ids]);unique=sorted(set(episodes.tolist()))
    energy=[]
    for i in ids:
        x=(data['features'][i].float()-pca['mean'][:,None,None,None])/pca['std'][:,None,None,None]
        energy.append(x.square().mean((0,2,3)).numpy())
    energy=np.asarray(energy);base_energy={'all':float(energy.mean()),'condition':float(energy[:,0].mean()),'pooled_target':float(energy[:,1:].mean())}
    block={'clips':len(data['manifest']),'test_episodes':len(unique),'test_clips':len(ids),'conditioning_target_energy':base_energy,'corruption_numeric_control':check,'methods':{},'paired_weight4_vs_weight1':{}}
    for label,names in [('pca48',['pca48']),('svae_w1',[f'svae-w1-s{s}' for s in SEEDS]),('svae_w4',[f'svae-w4-s{s}' for s in SEEDS])]:
        rr=[rows[k] for k in names];rec={m:stats([r['reconstruction'][m] for r in rr]) for m in ['all_mse','condition_mse','pooled_target_mse']}
        for group in base_energy:rec[group+'_relative_mse']=stats([r['reconstruction'][group+'_mse']/base_energy[group] for r in rr])
        block['methods'][label]={'reconstruction':rec,'probes':{variant:stats([r['probes'][variant]['normalized_mse'] for r in rr]) for variant in ['clean','noise10','brightness06']},'seeds_or_single':[{'condition':r['condition'],'clean_probe_mse':r['probes']['clean']['normalized_mse'],'noise_probe_mse':r['probes']['noise10']['normalized_mse'],'brightness_probe_mse':r['probes']['brightness06']['normalized_mse']} for r in rr]}
    for metric,cols in [('condition_mse',[0]),('pooled_target_mse',[1,2]),('all_mse',[0,1,2])]:
        vals=[]
        for w in [1,4]:
            vv=[]
            for seed in SEEDS:
                err=np.asarray(rows[f'svae-w{w}-s{seed}']['reconstruction']['per_clip'])[:,cols].mean(1)
                vv.append([float(err[episodes==e].mean()) for e in unique])
            vals.append(vv)
        block['paired_weight4_vs_weight1'][metric]=paired_ci(*vals)
    for variant in ['clean','noise10','brightness06']:
        vv=[[rows[f'svae-w{w}-s{s}']['probes'][variant]['episode_losses'] for s in SEEDS] for w in [1,4]]
        block['paired_weight4_vs_weight1'][variant+'_probe']=paired_ci(*vv)
    block['proprio_only']=rows['pca48']['proprio_only'];summary['tasks'][task]=block
    for path in folder.glob('svae-w*-s*/training.json'):
        tr=read(path);resources['training_seconds']+=tr['wall_seconds'];resources['peak_allocated_gb']=max(resources['peak_allocated_gb'],tr['peak_allocated_gb']);resources['trained_models']+=1
    resources['evaluation_seconds']+=sum(r['wall_seconds'] for r in rows.values())+check['wall_seconds']
    del data,pca
resources['initial_run_wall_seconds']=read(RUN/'last-run.json')['wall_seconds'];resources['matched_reevaluation_wall_seconds']=read(RUN/'corrections-complete.json')['wall_seconds'];summary['resources']=resources
summary['macro']={metric:stats([summary['tasks'][task]['paired_weight4_vs_weight1'][metric]['relative_change_percent'] for task in TASKS]) for metric in ['condition_mse','pooled_target_mse','clean_probe','noise10_probe','brightness06_probe']}
for v in summary['macro'].values():v['std_across_tasks']=v.pop('std_across_training_seeds')
summary['sampling_clarification']={'date': '2026-09-21', 'source': 'openwam/dataloader/robotwin.py:356-363', 'released_num_frames_means': 'raw HDF5 window length', 'raw_steps': 33, 'video_stride': 4, 'sampled_video_frames': 9, 'note': 'Earlier prose misread num_frames as the sampled frame count; all numerical experiment results unchanged.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
# Scientific figures with all fixed tasks and training-seed variation, no best-seed selection.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
labels=['Adjust bottle','Handover block','Object to basket'];colors=['#547257','#a9773c'];fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,(title,key,kind) in zip(axes,[('Current-frame reconstruction','condition_mse','reconstruction'),('Pooled-target reconstruction','pooled_target_mse','reconstruction'),('Clean state-change probe','clean','probes')]):
    x=np.arange(3)
    for offset,method,color in [(-.18,'svae_w1',colors[0]),(.18,'svae_w4',colors[1])]:
        stats_=[summary['tasks'][t]['methods'][method][kind][key] for t in TASKS]
        ax.bar(x+offset,[q['mean'] for q in stats_],.34,yerr=[q['std_across_training_seeds'] for q in stats_],color=color,label='Original loss' if method=='svae_w1' else 'Condition weight 4',capsize=3)
    ax.set_xticks(x,labels,rotation=15,ha='right');ax.set_title(title);ax.set_ylabel('MSE (lower is better)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Fixed 2,000-step budget · 3 training seeds · held-out episodes',fontsize=12)
fig.savefig(OUT/'paired-results.svg');fig.savefig(OUT/'paired-results.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,t,label in zip(axes,TASKS,labels):
    for method,color in [('pca48','#8396aa'),('svae_w1',colors[0]),('svae_w4',colors[1])]:
        vals=[summary['tasks'][t]['methods'][method]['probes'][v]['mean'] for v in ['clean','brightness06','noise10']]
        ax.plot(range(3),vals,'o-',label={'pca48':'PCA 48','svae_w1':'S-VAE original','svae_w4':'S-VAE weight 4'}[method],color=color)
    ax.set_xticks(range(3),['Clean','Brightness ×0.6','Noise σ=10/255'],rotation=15,ha='right');ax.set_title(label);ax.set_ylabel('Frozen clean-probe normalized MSE');ax.grid(axis='y',alpha=.15)
axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Synthetic input corruptions · matched encoder batch geometry',fontsize=12)
fig.savefig(OUT/'corruption-results.svg');fig.savefig(OUT/'corruption-results.png',dpi=180)
print(json.dumps({'macro':summary['macro'],'resources':resources},indent=2))
