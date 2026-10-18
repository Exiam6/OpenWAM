"""Descriptive, post-hoc gripper and tracking observations from finished episodes."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
rows=[]
for path in sorted((ROOT/'evaluation').glob('*/*/handover_block/seed-*/result.json')):
    result=json.loads(path.read_text())
    if result['status']!='complete':continue
    trace=[json.loads(line) for line in (path.parent/'trace.jsonl').read_text().splitlines()]
    actions=np.array([x['action'] for x in trace]);states=np.array([x['proprio'] for x in trace]);stats={}
    for side,start,gripper in [('left',0,9),('right',10,19)]:
        closed=np.flatnonzero(actions[:,gripper]<.5)
        error=np.linalg.norm(states[1:,start:start+3]-actions[:-1,start:start+3],axis=1)
        stats[side]=dict(command_min=float(actions[:,gripper].min()),first_closed_command_step=int(closed[0]) if len(closed) else None,
            closed_command_fraction=float(np.mean(actions[:,gripper]<.5)),median_target_to_next_observation_error_mm=float(1000*np.median(error)))
    rows.append(dict(cell=path.parents[3].name,condition=path.parents[2].name,scene_seed=result['scene_seed'],success=result['success'],steps=result['steps'],seconds=result['seconds'],grippers=stats))
(ROOT/'handover-gripper-observations.json').write_text(json.dumps(dict(
    scope='Post-hoc descriptive trace statistics for completed handover episodes only. Paired trajectories diverge after execution changes; not an additional hypothesis test.',rows=rows),indent=2)+'\n')
print(json.dumps(dict(finished_episodes=len(rows),fixed_successes=sum(x['success'] for x in rows if x['condition']=='fixed'))))
