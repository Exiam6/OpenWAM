"""Release fixed full training only from complete native preflight evidence."""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

import sys
sys.path.insert(0,'/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/code')
import joint_resources as j
from build_cache import sha

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')


def load(path):return json.loads(Path(path).read_text())


def nonnegative(value):
    assert isinstance(value,(int,float)) and math.isfinite(value) and value>=0
    return value


def readiness():
    plan=j.verify();j.legacy_failed();assignment=j.check_assignment(plan,require_live=False,check_window=False)
    registration=ROOT/'gpu-registration.json';assert registration.exists(),'CPU-only: no GPU registration'
    p=load(registration);assert p['gpu_launch_permitted'] and 'full_training' in p['released_stages']
    assert p['training_seconds_per_seed']==10800 and p['tiny_total_seconds']==1800 and p['cache_seconds_limit']==7200
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
    for path in p['pause_paths']:assert not Path(path).exists()
    for group in ['launcher_files','training_files','tiny_files','deployment_files','cache_files']:
        for path,digest in p[group].items():assert sha(path)==digest,path
    preflight_path=ROOT/'dedicated-preflight-run.json';preflight=load(preflight_path)
    assert preflight['resource_plan_sha256']==sha(j.PLAN) and preflight['assignment_sha256']==sha(j.ASSIGNMENT)
    assert preflight['status']=='complete' and preflight['passed'] and preflight['occupied_seconds']<9000
    assert preflight['cache_seconds']<7200 and preflight['tiny_combined_seconds']<1800
    assert [x['name'] for x in preflight['stages']]==['cache','training_smoke','deployment_contract','tiny_fit']
    assert all(x['execution']['complete'] and x['execution']['exit_code']==0 for x in preflight['stages'])
    cache_path=ROOT/'vae-cache/result.json';cache=load(cache_path)
    assert cache['passed'] and cache['completed']==6424 and cache['serialization_exact']
    assert sha(cache['cache'])==cache['sha256'] and cache['latents_shape']==[6424,48,3,24,20]
    tasks=set(p['scenes'])
    assert {x['task'] for x in cache['preflight']}==tasks and len(cache['preflight'])==3
    for row in cache['preflight']:
        for key in ['native_difference','first_frame_difference','future_to_first_difference']:assert nonnegative(row[key])<=1e-6
        assert nonnegative(row['future_difference'])>1e-6
    smoke_path=ROOT/'training-smoke/vae48_native-seed42/result.json';smoke=load(smoke_path)
    assert smoke['complete'] and smoke['steps']==3 and smoke['strict_reload'] and smoke['seed']==42
    assert sha(smoke['checkpoint'])==smoke['checkpoint_sha256']
    assert set(smoke['gradients'])=={'video_backbone.dit.patch_embedding.weight','video_backbone.dit.blocks.0.self_attn.q.weight','action_backbone.action_encoder.weight'}
    assert all(nonnegative(x)>0 for x in smoke['gradients'].values())
    contract_path=ROOT/'deployment-contract/result.json';contract=load(contract_path)
    assert contract['passed'] and contract['native_deployment_reload'] and contract['checkpoint_sha256']==smoke['checkpoint_sha256']
    assert len(contract['rows'])==3 and {x['task'] for x in contract['rows']}==tasks
    for row in contract['rows']:
        for key,limit in [('proprio_difference',1e-6),('latent_difference',1e-6),('action_difference',1e-5),('repeat_difference',1e-5)]:
            assert nonnegative(row[key])<=limit
        assert sha(row['trace'])==row['trace_sha256']
    tiny_path=ROOT/'tiny-fit/result.json';tiny=load(tiny_path)
    assert tiny['passed'] and tiny['steps']==512 and not tiny['checkpoint_reused_for_full_training']
    before,after=tiny['before'],tiny['after']
    for key in ['flow_action_loss','mean_generated_mse']:
        assert nonnegative(before[key])>0 and nonnegative(after[key])<=.25*before[key]
    assert len(before['rows'])==len(after['rows'])==3
    assert {x['task'] for x in before['rows']}=={x['task'] for x in after['rows']}==tasks
    for a,b in zip(before['rows'],after['rows']):
        assert (a['task'],a['cache_index'])==(b['task'],b['cache_index'])
        assert nonnegative(b['generated_action_mse'])<=.5*nonnegative(a['generated_action_mse'])
    projection=preflight['projection']
    assert projection['training_fits'] and nonnegative(projection['training_seconds_per_seed'])<10740
    assert projection['fixed_overhead_reserve_seconds']==900
    # Recompute from retained timing rows instead of accepting a passed flag alone.
    metrics=[json.loads(line) for line in (ROOT/'tiny-fit/metrics.jsonl').read_text().splitlines()]
    tail=metrics[-9:];assert len(tail)==9 and tail[-1]['step']==512
    rates=[(b['seconds']-a['seconds'])/(b['step']-a['step']) for a,b in zip(tail,tail[1:])]
    assert all(nonnegative(x)>0 for x in rates)
    measured=max(rates)*20000+900
    assert abs(measured-projection['training_seconds_per_seed'])<1e-6
    paths=[registration,preflight_path,cache_path,smoke_path,contract_path,tiny_path,ROOT/'tiny-fit/metrics.jsonl']
    return dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,preflight_state_path=str(preflight_path),joint_plan_path=str(j.PLAN),joint_plan_sha256=sha(j.PLAN),reserved_gpus=assignment['vae_gpus'],native_deployment_contract_passed=True,tiny_fit_passed=True,budget_projection_passed=True,
        native_deployment_contract_sha256=sha(contract_path),tiny_fit_sha256=sha(tiny_path),evidence_sha256={str(path):sha(path) for path in paths},
        planned_training_seeds=[42,43,44],fixed_training_steps=20000,training_seconds_per_seed=10800,projected_training_seconds_per_seed=measured,
        fresh_initialization_required=True,tiny_checkpoint_reused=False,
        scope='Preflight evidence and occupied-cost release only. The batch parent must independently recheck current dedicated GPU availability and full remaining allocation window before launching any seed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-readiness',action='store_true');args=parser.parse_args()
    try:result=readiness()
    except AssertionError as error:
        if args.check_readiness:print(str(error));raise SystemExit(2)
        raise
    if args.check_readiness:print(json.dumps(dict(ready=True,projected_training_seconds_per_seed=result['projected_training_seconds_per_seed'])))
    else:
        path=ROOT/'full-training-release.json'
        with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result))
