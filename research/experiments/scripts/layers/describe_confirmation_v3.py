"""Predeclared supplementary diagnostics after independent primary audit."""
from pathlib import Path
import datetime,json,os
import numpy as np,torch
R=Path(os.environ['LAYER_RUN']);O=R/'confirmation';tasks=['adjust_bottle','handover_block','place_object_basket']
assert json.loads((O/'audit.json').read_text())['passed'];assert not (O/'descriptive.json').exists(),'No completed supplementary replay'
rows=json.loads((O/'sample-manifest.json').read_text());d=np.load(O/'vectors.npz');pred=np.load(O/'predictions.npz');coef=np.load(R/'readout/coefficients.npz');mlp=torch.load(R/'readout/mlp/models.pt',map_location='cpu',weights_only=False);ref=d['reference'];condition=d['condition'];task=np.array([r['task'] for r in rows]);ep=np.array([r['seed'] for r in rows]);phase=np.max(np.abs(d['target'][:,6:]-d['state'][:,-2:]),axis=1)>.001
out={'scope':'supplementary descriptive analysis fixed after head-development results but before fresh scoring; not a primary-selection rule','created_at':datetime.datetime.now().astimezone().isoformat(),'gripper_change_predicate':'max(abs(target_gripper_at_rawframe_plus4-current_gripper)) > 0.001','contact_labels':False,'phase_counts':{},'phase_scores':{},'decomposition':{}}
for t in tasks:
    out['phase_counts'][t]={}
    for label,mask in [('change',phase),('stable',~phase)]:
        ii=(task==t)&mask;out['phase_counts'][t][label]={'frames':int(ii.sum()),'episodes':int(len(np.unique(ep[ii])))}
for key,p in pred.items():
    method,t,name,variant=key.split('/');ii=np.flatnonzero((task[ref]==t)&(condition==variant));rr=ref[ii];y=d['target'][rr]
    if method=='ridge':scale=np.r_[coef[f'{t}/{name}/0/ys'],coef[f'{t}/{name}/1/ys']];y=y.astype(float)
    else:scale=mlp['scales'][t+'/'+name][3].numpy()
    err=((p-y)/scale)**2;out['phase_scores'][key]={}
    for label,mask in [('change',phase[rr]),('stable',~phase[rr])]:
        ee=np.unique(ep[rr][mask]);vals=[[err[mask&(ep[rr]==e),:6].mean(),err[mask&(ep[rr]==e),6:].mean()] for e in ee]
        out['phase_scores'][key][label]={'episodes':ee.tolist(),'frames_before_noise_copies':int(out['phase_counts'][t][label]['frames']),'per_episode_groups':np.asarray(vals,dtype=float).tolist(),'E':float(np.mean(vals)) if vals else None,'translation':float(np.mean(np.array(vals)[:,0])) if vals else None,'gripper':float(np.mean(np.array(vals)[:,1])) if vals else None}
    if variant not in ['noise0.10','noise0.04']:continue
    clean_ii=np.flatnonzero((task[ref]==t)&(condition=='clean'));clean_ref=ref[clean_ii];mapping={v:i for i,v in enumerate(clean_ref)};clean=pred[f'{method}/{t}/{name}/clean'][[mapping[r] for r in rr]];p64=p.astype(np.float64);c64=clean.astype(np.float64);y64=y.astype(np.float64);s64=scale.astype(np.float64);e=(c64-y64)/s64;q=(p64-c64)/s64;increase=((p64-y64)/s64)**2-e*e;drift=q*q;cross=2*e*q
    assert np.allclose(drift+cross,increase,rtol=1e-10,atol=1e-12),key
    out['decomposition'][key]={}
    for label,array in [('clean_error',e*e),('noise_shift_squared',drift),('error_alignment_cross_term',cross),('error_increase',increase)]:out['decomposition'][key][label]=np.mean([[array[ep[rr]==j,:6].mean(),array[ep[rr]==j,6:].mean()] for j in np.unique(ep[rr])],axis=0).tolist()
(O/'descriptive.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'phase_counts':out['phase_counts'],'prediction_conditions':len(out['phase_scores']),'decomposition_identities_passed':len(out['decomposition'])},indent=2))
