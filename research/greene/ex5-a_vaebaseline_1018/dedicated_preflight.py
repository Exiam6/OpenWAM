"""Gated serial VAE preflight; no full training or evaluation is launched here."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0,'/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/code')
import joint_resources as j
import run_preflight as frozen
from build_cache import sha
from stage_runner import run_child

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
SOURCE=ROOT/'code'
ENGINE=Path('/home/zifanz4/openwam-rae-research')


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def admission():
    plan=j.verify();j.legacy_failed(before_gpu=True);j.check_assignment(plan)
    assert str(Path(__file__).resolve()) in plan['files']
    return frozen.admission()


def window(p,current=None):
    j.check_assignment(j.read(j.PLAN))
    return j.preflight_window(current=current)


def idle_gpus(p):
    assignment=j.check_assignment(j.read(j.PLAN))
    return list(j.capacity(assignment['vae_gpus'],p).values())


def main():
    p=admission();end=window(p);gpu=None
    lock=(ROOT/'preflight.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'dedicated-preflight-run.json';assert not path.exists(),'Do not retry or overwrite an earlier attempt'
    state=dict(pid=os.getpid(),status='checking_capacity',gpu=gpu,stages=[],started=dt.datetime.now(dt.timezone.utc).isoformat(),resource_plan_sha256=sha(j.PLAN),assignment_sha256=sha(j.ASSIGNMENT))
    write(path,state);started=None
    def stop(*_):raise RuntimeError('Preflight supervisor termination')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop)
    try:
        streak=0
        while True:
            end=window(p);samples=idle_gpus(p)
            available=sorted(x['gpu'] for x in samples if x['idle'])
            candidate=available[0] if available else None
            streak=streak+1 if candidate is not None and candidate==gpu else (1 if candidate is not None else 0)
            gpu=candidate
            state.update(status='waiting_capacity',gpu=gpu,capacity_sample=samples,admitted_samples=streak);write(path,state)
            if streak>=3:
                samples=idle_gpus(p)
                if next(x for x in samples if x['gpu']==gpu)['idle']:break
                streak=0
            time.sleep(5)
        state['capacity_sample']=samples;started=time.monotonic();signal.alarm(8990);state['status']='running';state['allocation_end']=end.isoformat();write(path,state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=p['gpus'][str(gpu)],PYTHONPATH=str(ENGINE),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        logs=ROOT/'dedicated-preflight-logs';logs.mkdir(exist_ok=False)
        def execute(name,script,args,limit,result_path):
            state['active_stage']=name;write(path,state)
            def record_child(pid):
                state['active_child_pid']=pid;write(path,state)
            execution=run_child([sys.executable,'-u',str(SOURCE/script),*args],log=logs/f'{name}.log',seconds=limit,deadline=end,pause_paths=p['pause_paths'],env=env,cwd=str(ENGINE),on_started=record_child)
            state['stages'].append(dict(name=name,execution=execution));state['active_child_pid']=None;write(path,state)
            assert execution['complete'] and not execution.get('error'),execution
            result=json.loads(result_path.read_text());assert result.get('passed',result.get('complete',False)),result
            return result
        cache=execute('cache','build_cache.py',[],7200,ROOT/'vae-cache/result.json')
        tiny_start=time.monotonic()
        smoke=execute('training_smoke','train_job.py',['--seed','42','--stage','smoke'],1800,ROOT/'training-smoke/vae48_native-seed42/result.json')
        def remaining_budget():
            prior=time.monotonic()-tiny_start
            assert prior<1740,'Insufficient combined tiny-stage budget remaining'
            return dict(prior_occupied_seconds=prior,remaining_seconds=1800-prior-1,smoke_result_sha256=sha(ROOT/'training-smoke/vae48_native-seed42/result.json'))
        budget=remaining_budget();write(ROOT/'deployment-budget.json',budget)
        contract=execute('deployment_contract','deployment_contract.py',[],budget['remaining_seconds'],ROOT/'deployment-contract/result.json')
        budget=remaining_budget();budget['deployment_result_sha256']=sha(ROOT/'deployment-contract/result.json');write(ROOT/'tiny-budget.json',budget)
        tiny=execute('tiny_fit','tiny_fit.py',[],budget['remaining_seconds'],ROOT/'tiny-fit/result.json')
        total_tiny=time.monotonic()-tiny_start;assert total_tiny<1800
        metrics=[json.loads(line) for line in (ROOT/'tiny-fit/metrics.jsonl').read_text().splitlines()]
        tail=metrics[-9:];assert len(tail)==9 and tail[-1]['step']==512
        # Training uses the same batch3 x accumulation2 path as this steady tiny-fit tail.
        rates=[(b['seconds']-a['seconds'])/(b['step']-a['step']) for a,b in zip(tail,tail[1:])]
        projected_training=max(rates)*20000+900
        projection=dict(max_tail_seconds_per_step=max(rates),training_seconds_per_seed=projected_training,fixed_overhead_reserve_seconds=900,
            training_fits=projected_training<10740,remaining_wall_seconds=(end-dt.datetime.now(dt.timezone.utc)).total_seconds(),
            endpoint_worst_remaining_seconds=21600,note='900s planning reserve covers initialization, two dev-loss passes and checkpoint I/O; measured feasibility is not a guaranteed runtime bound.')
        assert projection['training_fits'] and projection['remaining_wall_seconds']>22560,projection
        state.update(status='complete',passed=True,cache_seconds=state['stages'][0]['execution']['occupied_seconds'],tiny_combined_seconds=total_tiny,
            projection=projection,full_training_released=False,full_training_requires_separate_evidence_review=True)
    except BaseException as error:
        state.update(status='failed',passed=False,error=repr(error));raise
    finally:
        signal.alarm(0)
        if started is not None:state['occupied_seconds']=time.monotonic()-started
        state['finished']=dt.datetime.now(dt.timezone.utc).isoformat();write(path,state);lock.close()
    print(json.dumps(state))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-admission',action='store_true');args=parser.parse_args()
    if args.check_admission:admission();print('admitted with unchanged numerical stages')
    else:main()
