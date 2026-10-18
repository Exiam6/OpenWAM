"""Audit VAE rollouts against frozen scene controls and prior open-loop actions."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(path, historical, references):
    path=Path(path);row=json.loads(path.read_text())
    assert row['route']=='vae48_native' and row['status']=='complete'
    identity=(row['training_seed'],row['task'],row['scene_seed'])
    before=historical[identity]
    for key in ['initial_state','action_seed','step_limit','evaluation_instruction']:
        assert row[key]==before[key],(identity,key)
    cameras=row['initial_render_mae']
    assert set(cameras)=={'head_camera','left_camera','right_camera'}
    assert all(np.isfinite(x) and 0<=x<=1 for x in cameras.values())
    reference=references[f"{row['task']}|{row['scene_seed']}"]
    assert reference['training_seed']==row['training_seed'] and reference['action_seed']==row['action_seed']
    assert reference['task']==row['task'] and reference['scene_seed']==row['scene_seed']
    assert reference['initial_state']==row['initial_state'] and reference['instruction']==row['evaluation_instruction']
    trace=[]
    for line in (path.parent/'trace.jsonl').read_text().splitlines():
        item=json.loads(line);assert item['step']==len(trace)
        for key in ['action','proprio']:
            value=np.asarray(item[key]);assert value.shape==(20,) and np.isfinite(value).all()
        trace.append(item)
    assert len(trace)==row['steps'] and trace and trace[-1]['success']==row['success']
    count=min(8,len(trace));actual=np.array([x['action'] for x in trace[:count]])
    expected=np.array(reference['actions'][:count]);assert actual.shape==expected.shape
    diff=float(np.abs(actual-expected).max());assert diff<=1e-6,(identity,'first action chunk differs',diff)
    calls=[call for step in trace for call in step['planner_calls']]
    return dict(identity=['vae48_native',*identity],result=row,result_path=str(path),result_sha256=sha(path),trace_sha256=sha(path.parent/'trace.jsonl'),
        verified=True,first_chunk_actions=count,first_chunk_max_difference=diff,planner_calls=len(calls),planner_failures=sum(x['status']!='Success' for x in calls),
        reference_scope='Frozen PNG/state open-loop serving replay before scored rollouts; same checkpoint and server implementation, independent of simulator feedback. Not an independent policy implementation.')
