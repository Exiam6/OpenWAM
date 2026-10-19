"""Per-seed visual and measured-cost gate for the remaining45fixed evaluations."""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import statistics

from build_cache import sha
from make_pilot_pairs import readiness

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')


def estimate(rows,occupied):
    tasks={x['task'] for x in rows};assert len(rows)==15 and len(tasks)==3
    assert math.isfinite(occupied) and occupied>0
    for row in rows:
        assert math.isfinite(row['seconds']) and row['seconds']>0
        assert 0<row['steps']<=row['step_limit']
    records=[];episode_seconds=sum(x['seconds'] for x in rows)
    overhead=max(300.,occupied-episode_seconds)
    for task in sorted(tasks):
        selected=[x for x in rows if x['task']==task];assert len(selected)==5
        rates=[];setups=[]
        for row in selected:
            first=Path(row['result_path']).parent/'frames/0000-head_camera.jpg'
            last=first.parent/f"{row['steps']:04d}-head_camera.jpg"
            loop=last.stat().st_mtime-first.stat().st_mtime
            assert 0<loop<=row['seconds']+2,'Frame timing is not consistent with episode timing'
            rates.append(loop/row['steps']);setups.append(max(0,row['seconds']-loop))
        steps={x['step_limit'] for x in selected};assert len(steps)==1
        rate=statistics.median(rates);setup=max(setups)
        seconds=15*(setup+next(iter(steps))*rate*1.10)
        records.append(dict(task=task,remaining_episodes=15,native_step_limit=next(iter(steps)),median_loop_seconds_per_action=rate,
            max_setup_seconds=setup,loop_safety_factor=1.10,projected_remaining_seconds=seconds))
    return dict(tasks=records,overhead_seconds=overhead,projected_remaining_seconds=sum(x['projected_remaining_seconds'] for x in records)+overhead,
        method='Full native action cap for each of45remaining episodes, task-median action-loop wall rate plus10percent, maximum measured task setup, and max300s/measured job overhead. Partial episode durations are not extrapolated from success rates.')


def main(seed,visual_review,check):
    p,pairs=readiness(seed)
    pilot_path=ROOT/'evaluation'/f'seed{seed}'/'pilot/result.json';pilot=json.loads(pilot_path.read_text())
    claim_path=ROOT/'phase-claims'/f'pilot_evaluation-seed{seed}.json';claim=json.loads(claim_path.read_text())
    assert claim['complete'] and claim['exit_code']==0 and claim['result_sha256']==sha(pilot_path)
    assert claim['seed']==seed and Path(claim['result_path']).resolve()==pilot_path.resolve()
    assert claim['occupied_seconds']>=pilot['occupied_seconds']
    media_path=ROOT/f'pilot-media-seed{seed}.json';media=json.loads(media_path.read_text())
    assert media['seed']==seed and media['paired_videos']==15 and len(media['records'])==15
    review=json.loads(Path(visual_review).read_text());assert review['visually_inspected'] and review['paired_videos']==15
    assert set(review['tasks'])==set(p['scenes']) and len(review['records'])==3
    assert review['scope']=='six sampled action indices on the first scene of each task'
    expected={}
    for old_path,new_path in pairs:
        row=json.loads(new_path.read_text());expected[(row['task'],row['scene_seed'])]=(old_path.resolve(),new_path.resolve())
    seen=set()
    for record in media['records']:
        key=(record['task'],record['scene_seed']);assert key not in seen;seen.add(key)
        assert record['seed']==seed
        assert (Path(record['raw_result']).resolve(),Path(record['vae_result']).resolve())==expected[key]
        assert record['full_decode_passed'] and sha(record['video'])==record['video_sha256'] and sha(record['preview'])==record['preview_sha256']
        assert sha(record['raw_result'])==record['raw_result_sha256'] and sha(record['vae_result'])==record['vae_result_sha256']
    assert seen=={(task,scene) for task,scenes in p['scenes'].items() for scene in scenes[:5]}
    for task,scenes in p['scenes'].items():
        inspected=next(x for x in review['records'] if x['task']==task)
        assert inspected['scene_seed']==scenes[0]
        record=next(x for x in media['records'] if x['task']==task and x['scene_seed']==scenes[0])
        assert inspected['preview_sha256']==record['preview_sha256'] and inspected['indices']==record['preview_indices']
    rows=[]
    for _,path in pairs:
        row=json.loads(path.read_text());row['result_path']=str(path);rows.append(row)
    from resource_amendment import limit,check_remaining
    cost=estimate(rows,claim['occupied_seconds']);remaining=limit(p,seed)-claim['occupied_seconds']
    check_remaining(limit(p,seed),claim['occupied_seconds'],remaining,cost['projected_remaining_seconds'])
    assert math.isfinite(remaining) and cost['projected_remaining_seconds']<remaining-60,'Remaining fixed endpoint does not fit per-seed evaluation budget'
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
    for pause in p['pause_paths']:assert not Path(pause).exists()
    result=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,seed=seed,budget_projection_passed=True,pilot_result_sha256=sha(pilot_path),
        parent_claim_sha256=sha(claim_path),visual_review=str(Path(visual_review).resolve()),visual_review_sha256=sha(visual_review),media_sha256=sha(media_path),cost=cost,
        remaining_episodes=45,prior_occupied_seconds=claim['occupied_seconds'],remaining_seconds=remaining,
        resource_authorization=p['resource_authorization'],resource_authorization_sha256=p['resource_authorization_sha256'],evaluation_limit_seconds=limit(p,seed),
        scope='Runtime/visual gate only. Successes and failures both remain; no checkpoint or scene selection. Resource parent rechecks fresh availability and allocation time.')
    if check:print(json.dumps(dict(ready=True,seed=seed,projected_remaining_seconds=cost['projected_remaining_seconds'],remaining_seconds=remaining)));return
    budget=ROOT/f'evaluation-budget-seed{seed}.json';release=ROOT/f'evaluation-release-seed{seed}.json'
    assert not budget.exists() and not release.exists(),'Preserve prior release attempt'
    with budget.open('x') as f:json.dump(dict(pilot_result_sha256=result['pilot_result_sha256'],prior_occupied_seconds=claim['occupied_seconds'],remaining_seconds=remaining),f,indent=2)
    with release.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,43,44],required=True);parser.add_argument('--visual-review',required=True);parser.add_argument('--check-readiness',action='store_true');args=parser.parse_args()
    try:main(args.seed,args.visual_review,args.check_readiness)
    except AssertionError as error:
        if args.check_readiness:print(str(error));raise SystemExit(2)
        raise
