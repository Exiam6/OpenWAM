"""Descriptive error decomposition, phase counts and all existing controls."""
from pathlib import Path
import json,numpy as np
R=Path('/home/zifanz4/openwam-experiments/studies/layers-20260922/development')
rows=json.loads((R/'manifest.json').read_text());data=np.load(R/'audit-labels.npz');coef=np.load(R/'coefficients.npz');task=np.array([r['task'] for r in rows]);ep=np.array([r['episode'] for r in rows]);tr=np.array([r['split']=='train' for r in rows]);tasks=['adjust_bottle','handover_block','place_object_basket'];ref=data['noisy_reference'];noise=data['noisy_condition'];target=data['target'];state=data['state'] if 'state' in data else None
out={'scope':'post-development descriptive decomposition; no new fitting or fresh scoring','groups':['translation','gripper'],'decomposition':{},'controls':{}}
for method,folder in [('ridge',R),('mlp',R/'mlp')]:
    predictions=np.load(folder/'predictions.npz');summ=json.loads((folder/('development-summary.json' if method=='ridge' else 'summary.json')).read_text());out['decomposition'][method]={};out['controls'][method]={}
    names=sorted({k.split('/')[1] for k in predictions.files})
    for name in names:
        task_records=[]
        for t in tasks:
            va=np.flatnonzero((task==t)&~tr);ii=np.flatnonzero((task[ref]==t)&(noise=='noise0.10'));rr=ref[ii];clean=predictions[t+'/'+name+'/clean'];pos={v:i for i,v in enumerate(va)};aligned=clean[[pos[i] for i in rr]];noisy=predictions[t+'/'+name+'/noise0.10'];scale=np.concatenate([coef[f'{t}/{name}/{g}/ys'] for g in [0,1]]) if method=='ridge' else target[(task==t)&tr].std(0).clip(1e-6)
            y=target[rr].astype(float) if method=='ridge' else target[rr];e=(aligned-y)/scale;q=(noisy-aligned)/scale;drift=q*q;cross=2*e*q;change=((noisy-y)/scale)**2-e*e
            assert np.allclose(drift+cross,change,rtol=2e-5,atol=2e-6),(method,t,name)
            record={'task':t}
            for label,array in [('clean_error',e*e),('noise_shift_squared',drift),('error_alignment_cross_term',cross),('error_increase',change)]:record[label]=np.mean([[array[ep[rr]==j,:6].mean(),array[ep[rr]==j,6:].mean()] for j in np.unique(ep[rr])],axis=0).tolist()
            task_records.append(record)
        out['decomposition'][method][name]={'per_task':task_records,'task_mean':{k:np.mean([r[k] for r in task_records],axis=0).tolist() for k in ['clean_error','noise_shift_squared','error_alignment_cross_term','error_increase']}}
        conditions=['clean','noise0.10']+(['spatial_shuffle','visual_mismatch'] if method=='ridge' and name!='proprio' else [])
        out['controls'][method][name]={c:{'E':float(np.mean([summ['conditions'][t+'/'+name+'/'+c]['E'] for t in tasks])),'translation':float(np.mean([summ['conditions'][t+'/'+name+'/'+c]['translation'] for t in tasks])),'gripper':float(np.mean([summ['conditions'][t+'/'+name+'/'+c]['gripper'] for t in tasks]))} for c in conditions}
out['label_ranges']={'translation_min':target[:,:6].min(0).tolist(),'translation_max':target[:,:6].max(0).tolist(),'future_gripper_min':target[:,6:].min(0).tolist(),'future_gripper_max':target[:,6:].max(0).tolist()}
if state is not None:
    phase=np.max(np.abs(target[:,6:]-state[:,-2:]),axis=1)>.001;out['phase_counts_development']={t:{'change_frames':int(np.sum((task==t)&~tr&phase)),'stable_frames':int(np.sum((task==t)&~tr&~phase)),'change_episodes':len(np.unique(ep[(task==t)&~tr&phase])),'stable_episodes':len(np.unique(ep[(task==t)&~tr&~phase]))} for t in tasks}
(R/'descriptive.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'scope':out['scope'],'label_ranges':out['label_ranges'],'phase_counts':out.get('phase_counts_development','state not cached')},indent=2))
