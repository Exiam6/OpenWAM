"""Training-only feasibility audit for a future visually conditioned probe.

No images, development/test episodes, models or performance scores are read.
This does not replace the current frozen primary comparison.
"""
from pathlib import Path
import datetime,json
import h5py,numpy as np
B=Path('/home/zifanz4/openwam-experiments');S=B/'studies/layers-20260922';R=Path('/data02/zifanz4/openwam-experiments/results/layers-20260922/native-preflight/train-data');manifest=json.loads((S/'development/manifest.json').read_text());allowed={(r['task'],r['episode']) for r in manifest if r['split']=='train'};records=[];summary={}
for task in ['adjust_bottle','handover_block','place_object_basket']:
    initial=[];waypoints=[];arms=[];times=[];paths=sorted((R/task).rglob('*.hdf5'));assert len(paths)==35
    for p in paths:
        episode=int(p.stem.replace('episode',''));assert (task,episode) in allowed
        with h5py.File(p,'r') as f:
            grip=np.stack([f['endpose/left_gripper'][:],f['endpose/right_gripper'][:]],axis=1);initial.append(np.r_[f['endpose/left_endpose'][0],f['endpose/right_endpose'][0],grip[0]])
            # Explicitly a numeric downward gripper crossing, not a contact event.
            candidates=[]
            for ai,arm in enumerate(['left','right']):
                indices=np.flatnonzero((grip[:-1,ai]>.5)&(grip[1:,ai]<=.5))+1
                if len(indices):candidates.append((int(indices[0]),ai,arm))
            rec={'task':task,'episode':episode,'first_gripper':grip[0].tolist(),'valid':bool(candidates)}
            if candidates:
                frame,ai,arm=min(candidates);position=f[f'endpose/{arm}_endpose'][frame,:3];rec.update(event_frame=frame,arm=arm,native_endpose_position=position.tolist(),simultaneous_crossing=len(candidates)>1 and candidates[0][0]==candidates[1][0]);waypoints.append(position);arms.append(ai);times.append(frame)
            records.append(rec)
    x=np.stack(initial);y=np.stack(waypoints) if waypoints else np.empty((0,3));summary[task]={'train_episodes':35,'valid_first_downward_crossing':len(waypoints),'initial_proprio_std':x.std(0).tolist(),'initial_proprio_range':np.ptp(x,axis=0).tolist(),'event_position_std_native':y.std(0).tolist() if len(y) else None,'event_position_range_native':np.ptp(y,axis=0).tolist() if len(y) else None,'event_frame_min_max':[min(times),max(times)] if times else None,'left_events':arms.count(0),'right_events':arms.count(1)}
out={'checked_at':datetime.datetime.now().astimezone().isoformat(),'scope':'training-only label/state structural audit for a possible future probe; not a trained/evaluated new method','predicate':'earliest downward native gripper crossing0.5 across botharms','contact_label_claim':False,'future_task_definition_not_finalized':True,'dev_or_confirmation_read':False,'summary':summary,'records':records};(S/'initial-goal-label-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
