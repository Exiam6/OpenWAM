"""Post-hoc, all-route input shift decomposition on saved vectors; no refits."""
from pathlib import Path
import json,datetime
import numpy as np
R=Path('/data02/zifanz4/openwam-experiments/results/rae-baseline-20260922');L=R.parent/'layers-20260922'
plan={'recorded_before_diagnostic':datetime.datetime.now().astimezone().isoformat(),'scope':'Post-hoc explanation of unusually large noise errors. All routes/tasks/noise levels, no selected examples, no benefit claim.','measures':['training feature std distribution','clean/noisy standardized feature RMS','paired noise feature delta RMS and mean-shift energy fraction','noise-induced normalized output shift and mean-bias energy fraction from fixed linear coefficients'],'prohibition':'No encoder inference, refitting, corrected test scores, heldout-statistic calibration, or change to original result.'}
assert not (R/'shift-diagnostic.json').exists();(R/'shift-diagnostic-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
wan=np.load(R/'wan-evaluation-vectors.npz');dino=np.load(L/'confirmation/vectors.npz');wc=np.load(R/'wan-coefficients.npz');sc=np.load(L/'readout/coefficients.npz');gc=np.load(L/'initial-goal/coefficients.npz');rows=json.loads((L/'confirmation/sample-manifest.json').read_text());task=np.array([x['task'] for x in rows]);frame=np.array([x['frame'] for x in rows]);out={}
for name in ['Wan48_native_scale','Wan48_per_token_layernorm','L12_raw','L12_pca','L12_svae42','L12_svae43','L12_svae44']:
 data=wan if name.startswith('Wan') else dino;ref=data['reference'];cond=data['condition'];cleanids=np.flatnonzero(cond=='clean');assert np.array_equal(ref[cleanids],np.arange(720));clean=data[name][cleanids].astype(float)
 for endpoint in ['state','goal']:
  for t in ['adjust_bottle','handover_block','place_object_basket']:
   if name.startswith('Wan'):cf=wc;key=f'state/{name}/{t}/translation' if endpoint=='state' else f'goal/{name}/{t}'
   else:cf=sc if endpoint=='state' else gc;key=f'{t}/{name}/0' if endpoint=='state' else f'{t}/{name}'
   xm=cf[key+'/xm'][:clean.shape[1]];xs=cf[key+'/xs'][:clean.shape[1]];w=cf[key+'/w'][:clean.shape[1]];totaldim=len(cf[key+'/xm'])
   if endpoint=='state':
    key2=key.rsplit('/',1)[0]+'/gripper' if name.startswith('Wan') else f'{t}/{name}/1';assert np.array_equal(xs,cf[key2+'/xs'][:clean.shape[1]]);w=np.column_stack([w,cf[key2+'/w'][:clean.shape[1]]])
   for condition in ['noise0.10','noise0.04']:
    ii=np.flatnonzero((task[ref]==t)&(cond==condition)&((frame[ref]==0) if endpoint=='goal' else True));rr=ref[ii];x=data[name][ii].astype(float);base=clean[rr];dx=(x-base)/xs;z=(x-xm)/xs;cz=(base-xm)/xs;proj=dx/np.sqrt(totaldim)@w
    def stats(a):
     energy=float(np.square(a).mean());bias=float(np.square(a.mean(0)).mean());return {'RMS':float(np.sqrt(energy)),'mean_shift_energy_fraction':bias/energy if energy>0 else None}
    out['/'.join([endpoint,name,t,condition])]={'training_std_min_median_max':np.quantile(xs,[0,.5,1]).tolist(),'training_std_floor_count':int((xs<=1e-6).sum()),'clean_train_standardized_RMS':float(np.sqrt((cz*cz).mean())),'noisy_train_standardized_RMS':float(np.sqrt((z*z).mean())),'paired_standardized_feature_delta':stats(dx),'paired_normalized_output_shift':stats(proj),'output_note':'state concatenates6translation and2gripper coordinates for this diagnostic; published state score weights the2groups equally'}
result={'scope':plan['scope'],'conditions':out,'measures_only_saved_vectors_and_coefficients':True,'no_correction_or_fit':True};(R/'shift-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
for ep in ['state','goal']:
 for name in ['Wan48_native_scale','Wan48_per_token_layernorm','L12_raw','L12_pca','L12_svae42']:
  a=[out[f'{ep}/{name}/{t}/noise0.10'] for t in ['adjust_bottle','handover_block','place_object_basket']]
  print(ep,name,json.dumps({'clean_z_RMS':[x['clean_train_standardized_RMS'] for x in a],'noisy_z_RMS':[x['noisy_train_standardized_RMS'] for x in a],'output_bias_fraction':[x['paired_normalized_output_shift']['mean_shift_energy_fraction'] for x in a]}))
