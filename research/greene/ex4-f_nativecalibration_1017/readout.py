"""Report finished pilot scenes without treating pending scenes as failures."""
import datetime as dt
import json
from pathlib import Path

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
P=json.loads((ROOT/'registration.json').read_text())
baseline=json.loads((ROOT/'baseline-selected.json').read_text())
rows=[]
for variant in P['variants']:
    for seed in P['seeds']:
        for task,seeds in P['selected'].items():
            dest=ROOT/'evaluation'/f'{variant}-seed{seed}'/'fixed'/task
            results=[]
            for scene in seeds:
                path=dest/f'seed-{scene}'/'result.json'
                if path.exists():results.append(json.loads(path.read_text()))
            before={x['scene_seed']:x for x in baseline if x['route']==variant and x['training_seed']==seed and x['task']==task}
            complete=[x for x in results if x['status']=='complete']
            rows.append(dict(variant=variant,seed=seed,task=task,planned=len(seeds),finished=len(complete),
                infrastructure_failures=len(results)-len(complete),successes=sum(x['success'] for x in complete),
                baseline_on_finished=sum(before[x['scene_seed']]['success'] for x in complete),
                wins=sum(x['success'] and not before[x['scene_seed']]['success'] for x in complete),
                losses=sum(not x['success'] and before[x['scene_seed']]['success'] for x in complete)))
done=all(x['finished']==x['planned'] for x in rows)
controls=[]
for path in (ROOT/'evaluation').glob('*/old/handover_block/summary.json'):
    d=json.loads(path.read_text());controls.append(dict(cell=path.parents[2].name,complete=d['complete'],successes=d['successes'],episodes=d['episodes']))
done=done and len(controls)==6 and all(x['complete'] and x['successes']==0 for x in controls)
arms={variant:sum(x['successes'] for x in rows if x['variant']==variant and x['task']=='handover_block') for variant in P['variants']}
verdict='pending'
if done:verdict='pilot_transfer_supported' if max(arms.values())>=2 else ('no_handover_success_on_pilot' if max(arms.values())==0 else 'undecided')
result=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),complete=done,verdict=verdict,rows=rows,old_controls=controls,
    note='Development pilot on first five frozen accepted scenes; corrected execution protocol, no RAE ranking claim.')
(ROOT/'pilot-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=done,verdict=verdict,finished=sum(x['finished'] for x in rows),planned=90,handover_successes=arms)))
