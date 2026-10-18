"""Independent post-validation, including all cameras and paired first actions."""
import datetime as dt
import json
from pathlib import Path

import numpy as np

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
BASE=ROOT.parent/'rae-factorial-20261010'
P=json.loads((ROOT/'registration.json').read_text())

def trace(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    rows=[];issues=[]
    for path in sorted((ROOT/'evaluation').glob('*/*/*/seed-*/result.json')):
        result=json.loads(path.read_text());cell=path.parents[3].name;condition=path.parents[2].name;task=result['task'];seed=result['scene_seed']
        row=dict(cell=cell,condition=condition,task=task,scene_seed=seed,status=result['status'])
        try:
            assert result['status']=='complete',result.get('error')
            assert seed in P['selected'][task]
            errors=result['initial_render_mae'];assert set(errors)=={'head_camera','left_camera','right_camera'}
            assert all(value<=1 for value in errors.values()),errors
            current=trace(path.parent/'trace.jsonl')
            assert len(current)==result['steps']
            assert [x['step'] for x in current]==list(range(result['steps']))
            assert all(np.asarray(x['action']).shape==(20,) and np.isfinite(x['action']).all() for x in current)
            assert bool(current[-1]['success'])==result['success']
            old_dir=BASE/'evaluation/fresh'/cell/task/'clean'/f'seed-{seed}'
            old_result=json.loads((old_dir/'result.json').read_text());old=trace(old_dir/'trace.jsonl')
            assert result['initial_state']==old_result['initial_state']
            assert result['action_seed']==old_result['action_seed']
            assert result['step_limit']==old_result['step_limit']
            assert result['evaluation_instruction']==old_result['evaluation_instruction']
            first=min(8,len(current),len(old))
            diff=float(np.max(np.abs(np.array([x['action'] for x in current[:first]])-np.array([x['action'] for x in old[:first]]))))
            calls=[call for x in current for call in x['planner_calls']]
            row.update(verified=True,initial_state_exact=True,all_three_camera_mae=errors,
                paired_first_chunk_max_action_difference=diff,first_chunk_actions=first,
                old_success=old_result['success'],success=result['success'],seconds=result['seconds'],old_seconds=old_result['seconds'],
                planner_calls=len(calls),planner_failures=sum(x['status']!='Success' for x in calls),
                physics_ticks=current[-1]['physics_ticks'])
        except Exception as exc:
            row.update(verified=False,error=repr(exc));issues.append(dict(path=str(path),error=repr(exc)))
        rows.append(row)
    output=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),verified_finished_episodes=bool(rows) and not issues,
        inspected=len(rows),issues=issues,rows=rows,
        scope='Only written per-scene results; pending episodes are not checked or counted as failures. First-action differences are reported, not silently thresholded.')
    (ROOT/'independent-result-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(inspected=len(rows),issues=issues,max_first_chunk_difference=max((x.get('paired_first_chunk_max_action_difference',0) for x in rows),default=None))))

if __name__=='__main__':main()
