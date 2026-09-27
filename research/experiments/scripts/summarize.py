import json,statistics,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/vol13/zifanz4/openwam-experiments');D=ROOT/'results';P=ROOT/'public-results';P.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
base=D/'pilot-adjust-bottle-v1';m=read(base/'metrics.json');e=read(base/'extraction.json')
runs=[]
for seed in [42,43,44]:
 r=D/f'pilot-adjust-bottle-s{seed}-2000'
 assert read(r/'last-run.json')['stage']=='fit'
 runs.append({'seed':seed,'metrics':read(r/'metrics.json'),'training':read(r/'training.json'),'curve':read(r/'training-curve.json'),'run':read(r/'last-run.json')})
def row(metrics,name):return next(x for x in metrics['reconstruction']if x['condition']==name)
def probe(metrics,name):return next(x for x in metrics['probes']if x['condition']==name)
rows=[]
for name,label in [('raw','Raw DINOv3'),('pca24','PCA-24'),('pca48','PCA-48'),('pca96','PCA-96'),('svae_fresh48','S-VAE-48 / 500 steps')]:
 a=row(m,name);b=probe(m,name+'_visual_plus_proprio')
 rows.append({'label':label,'feature_mse':a['test_standardized_feature_mse'],'probe_mse':b['normalized_mse'],'translation_rmse_mm':b['translation_rmse_mm'],'gripper_rmse':b['gripper_rmse'],'feature_mse_std':0,'probe_mse_std':0,'training_seeds':1,'channels':a['channels'],'episode_bootstrap_95ci':b['normalized_mse_episode_bootstrap_95ci']})
fresh=[]
for r in runs:
 a=row(r['metrics'],'svae_fresh48');b=probe(r['metrics'],'svae_fresh48_visual_plus_proprio')
 fresh.append({'seed':r['seed'],'feature_mse':a['test_standardized_feature_mse'],'probe_mse':b['normalized_mse'],'translation_rmse_mm':b['translation_rmse_mm'],'gripper_rmse':b['gripper_rmse'],'selected_step':r['metrics']['selected_svae_step'],'episode_bootstrap_95ci':b['normalized_mse_episode_bootstrap_95ci']})
agg={'label':'S-VAE-48 / 2000 steps','channels':48,'training_seeds':3}
for key in ['feature_mse','probe_mse','translation_rmse_mm','gripper_rmse']:
 vals=[x[key]for x in fresh];agg[key]=statistics.mean(vals);agg[key+'_std']=statistics.stdev(vals)
rows.append(agg)
for r in rows:
 for k in ['feature_mse','probe_mse','translation_rmse_mm','gripper_rmse']:assert math.isfinite(r[k])
summary={'title':'OpenWAM offline representation pilot','date':'2026-09-20','status':'completed','closed_loop_evaluated':False,'dataset':{'name':'RoboTwin2.0 adjust_bottle clean50','episodes':50,'clips':600,'train_episodes':35,'validation_episodes':5,'test_episodes':10,'train_clips':420,'validation_clips':60,'test_clips':120,'split_seed':42,'resolution':[384,320],'video_frames':9,'raw_window_steps':33,'video_stride':4},'gpu':{'model':'NVIDIA L40S','count':1,'torch': '2.7.0+cu126','transformers':'5.17.0','max_pytorch_allocated_gb':max([read(base/'training.json')['peak_allocated_gb']]+[r['training']['peak_allocated_gb']for r in runs]),'total_job_wall_seconds':read(base/'last-run.json')['wall_seconds']+sum(r['run']['wall_seconds']for r in runs)},'extraction':{k:e[k]for k in ['extract_wall_seconds','median_encoder_seconds_per_clip','feature_cache_bytes','condition_future_invariance_max_abs']},'rows':rows,'fresh_svae_2000_by_seed':fresh,'proprio_only':probe(m,'proprio_only'),'simple_baselines':m['baselines'],'published_svae_reference':{'feature':row(m,'svae_published48'),'probe':probe(m,'svae_published48_visual_plus_proprio'),'independent_holdout':False},'target':'Four-raw-step future achieved end-effector XYZ displacement and achieved gripper state; current visual observation plus proprioception input; not controller action or policy success.','limitations':['One simulated task, fixed episode split, correlated windows within episodes.','Ridge penalty and S-VAE checkpoint selected on validation, test used for reporting.','The three-seed follow-up was specified after the 500-step pilot; all are exploratory results.','2000-step S-VAE is a fixed-budget study, not proof of convergence.','Published S-VAE may have seen all pilot trajectories and is excluded from the fair comparison.','Stronger feature reconstruction does not establish higher closed-loop success.'],'source':read(ROOT/'assets/dinov3-study/provenance.json')}
summary['gpu']['total_gpu_hours_upper_bound_one_reserved_gpu']=summary['gpu']['total_job_wall_seconds']/3600
(P/'pilot-summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f5ef','axes.facecolor':'#f6f5ef','axes.edgecolor':'#c6cfc1','text.color':'#223b31','axes.labelcolor':'#223b31','xtick.color':'#223b31','ytick.color':'#223b31'})
show=rows[1:];labels=['PCA\n24D','PCA\n48D','PCA\n96D','S-VAE 48D\n500 steps','S-VAE 48D\n2000 steps'];colors=['#a9b992','#879f64','#627f4a','#b4cc56','#234b38']
fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for ax,key,title in zip(axs,['feature_mse','probe_mse'],['Feature reconstruction','Future state-change probe']):
 vals=[r[key]for r in show];errs=[r[key+'_std']for r in show]
 ax.bar(np.arange(len(show)),vals,color=colors,width=.65)
 ax.errorbar(len(show)-1,vals[-1],yerr=errs[-1],fmt='none',color='#223b31',capsize=4)
 ax.set_xticks(np.arange(len(show)),labels);ax.set_ylabel('Standardized MSE (lower is better)');ax.set_title(title,loc='left',fontweight='bold',pad=15)
 for i,v in enumerate(vals):ax.text(i,v+errs[i]+max(vals)*.045,f'{v:.3f}',ha='center',fontsize=10)
 ax.set_ylim(0,max(vals)*1.28);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
axs[1].axhline(summary['proprio_only']['normalized_mse'],color='#9a754a',ls='--',lw=1,label='Proprioception only');axs[1].set_ylim(0,max(summary['proprio_only']['normalized_mse']*1.2,axs[1].get_ylim()[1]));axs[1].legend(frameon=False,fontsize=8)
fig.suptitle('OpenWAM representation pilot | 50 episodes, 1 task, 1 L40S',fontsize=14,fontweight='bold',x=.03,ha='left')
fig.text(.5,-.035,'S-VAE 2000-step error bars: standard deviation over 3 training seeds. Not policy success rates.',ha='center',fontsize=9)
fig.savefig(P/'compression-results.svg',bbox_inches='tight');fig.savefig(P/'compression-results.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.6),layout='constrained')
for r,c in zip(runs,['#264f3b','#70954d','#a29b47']):ax.plot([x['step']for x in r['curve']],[x['val_mse']for x in r['curve']],label=f'Seed {r["seed"]}',color=c,lw=2)
ax.set(xlabel='Optimizer steps',ylabel='Validation feature MSE',title='Fresh S-VAE-48: fixed 2000-step training budget');ax.legend(frameon=False);ax.grid(alpha=.15);fig.savefig(P/'training-curves.svg',bbox_inches='tight');plt.close(fig)
print(json.dumps({'rows':rows,'gpu':summary['gpu'],'seeds':fresh},indent=2))
