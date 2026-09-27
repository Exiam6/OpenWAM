#!/usr/bin/env python3
"""Post-hoc diagnostic: independent group-specific ridge regularization.

No new representation training. This does not replace the predeclared joint
probe results, and it does not reselect the auxiliary coefficient.
"""
import json
import time
import numpy as np
import torch
import night as n
p=n.p
RUN=p.ROOT/'results/control-20260921'
OUT=p.ROOT/'public-results/control-20260921'

def main():
    start=time.perf_counter()
    result={'status':'post_hoc_diagnostic_after_primary_results','auxiliary_coefficient_reselected':False,
            'representation_training_runs':0,'selection':'Ridge alpha selected on existing validation episodes independently per output group',
            'primary_joint_probe_unchanged':True,'tasks':{}}
    for task in n.TASKS:
        data=torch.load(n.RUNROOT/task/'features.pt',mmap=True,map_location='cpu',weights_only=False)
        state=data['state'].numpy();te=p.ids(data,'test').numpy()
        episodes=np.asarray([data['manifest'][i]['episode'] for i in te])
        rows={}
        names=['raw']+[f'aux{weight}-s{seed}' for weight in ['0','1'] for seed in n.SEEDS]
        for name in names+['proprio_only']:
            if name=='proprio_only':x=state
            else:x=np.concatenate([np.load(RUN/task/name/'vectors.npy'),state],axis=1)
            rows[name]={}
            for group,cols,scale in [('translation',slice(0,6),1000.),('gripper',slice(6,8),1.)]:
                part=dict(data);part['target']=data['target'][:,cols]
                probe=n.fit_probe(x,part);pred=n.predict(probe,x[te]);target=part['target'].numpy()[te]
                err=pred-target;normalized=(err/probe['ys'])**2
                rows[name][group]={'alpha':probe['alpha'],'validation_normalized_mse':float(probe['val_mse']),
                    'test_normalized_mse':float(normalized.mean()),'test_rmse':float(scale*np.sqrt(np.mean(err**2))),
                    'rmse_units':'mm' if group=='translation' else 'dataset gripper-state units',
                    'episode_ids':sorted(set(episodes.tolist())),
                    'episode_losses':[float(normalized[episodes==ep].mean()) for ep in sorted(set(episodes))]}
        result['tasks'][task]={'methods':rows,'paired_mean_rmse_changes':{}}
        for group in ['translation','gripper']:
            a=np.mean([rows[f'aux0-s{seed}'][group]['test_rmse'] for seed in n.SEEDS])
            b=np.mean([rows[f'aux1-s{seed}'][group]['test_rmse'] for seed in n.SEEDS])
            result['tasks'][task]['paired_mean_rmse_changes'][group]={'baseline':float(a),'candidate':float(b),'relative_change_percent':float(100*(b/a-1))}
        print(task,json.dumps(result['tasks'][task]['paired_mean_rmse_changes']),flush=True)
    result['wall_seconds']=time.perf_counter()-start
    (OUT/'group-probe-diagnostic.json').write_text(json.dumps(result,indent=2))
    print('GROUP PROBE COMPLETE',flush=True)

if __name__=='__main__':main()
