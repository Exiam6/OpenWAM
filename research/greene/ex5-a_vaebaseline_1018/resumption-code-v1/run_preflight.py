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

from build_cache import sha
from stage_runner import run_child

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
SOURCE=Path(__file__).parent
ENGINE=Path('/home/zifanz4/openwam-rae-research')
EXTRAS=(0,5,6,7)


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def admission():
    path=ROOT/'gpu-registration.json';assert path.exists(),'CPU-only: GPU stage is not registered'
    p=json.loads(path.read_text());assert p['gpu_launch_permitted'] and p['preflight_launch_permitted']
    assert p['cache_seconds_limit']==7200 and p['tiny_total_seconds']==1800
    for name,digest in p['launcher_files'].items():assert sha(name)==digest,name
    assert str(Path(__file__).resolve()) in p['launcher_files'] and str(SOURCE/'stage_runner.py') in p['launcher_files']
    for name,digest in p['predecessor_files'].items():assert sha(name)==digest,name
    for name,digest in p['asset_files'].items():assert sha(name)==digest,name
    h=ROOT.parent/'ex4-h_officialcalibration_1017';g=ROOT.parent/'ex4-g_calibratedfactorial_1017'
    summary=json.loads((h/'success-summary.json').read_text());bundle=json.loads((h/'final-bundle.json').read_text())
    assert summary['complete'] and summary['verdict']=='predicted_calibration_transfer_supported'
    assert bundle['complete'] and bundle['pairs']==15 and sha(bundle['path'])==bundle['sha256']
    assert json.loads((g/'L3-release.json').read_text())['passed']
    for pause in p['pause_paths']:assert not Path(pause).exists()
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
    return p


def window(p,current=None):
    current=current or dt.datetime.now(dt.timezone.utc)
    start=dt.datetime.fromisoformat(p['extra_window_start'])
    end=min(dt.datetime.fromisoformat(p['extra_window_end']),dt.datetime.fromisoformat(p['deadline']))-dt.timedelta(seconds=60)
    # Preserve the full endpoint: worst-case cache+tiny, then three parallel training/evaluation jobs.
    assert start<=current and (end-current).total_seconds()>7200+1800+10800+10800+900+60
    return end


def idle_gpus(p):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
    occupied={line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    samples=[]
    for line in output.splitlines():
        index,uuid,free,total,util=[x.strip() for x in line.split(',')];index=int(index)
        if index not in EXTRAS:continue
        assert uuid==p['gpus'][str(index)],'Physical GPU identity changed'
        samples.append(dict(gpu=index,uuid=uuid,free_mib=int(free),util=int(util),idle=uuid not in occupied and int(util)==0 and int(total)-int(free)<256))
    return samples


def main(gpu):
    p=admission();assert gpu in EXTRAS;end=window(p)
    lock=(ROOT/'preflight.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'preflight-run.json';assert not path.exists(),'Do not retry or overwrite an earlier attempt'
    state=dict(pid=os.getpid(),status='checking_capacity',gpu=gpu,stages=[],started=dt.datetime.now(dt.timezone.utc).isoformat())
    write(path,state);started=None
    def stop(*_):raise RuntimeError('Preflight supervisor termination')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop)
    try:
        streak=0
        while True:
            end=window(p);samples=idle_gpus(p)
            eligible=sum(x['idle'] for x in samples)>=3 and next(x for x in samples if x['gpu']==gpu)['idle']
            streak=streak+1 if eligible else 0
            state.update(status='waiting_capacity',capacity_sample=samples,admitted_samples=streak);write(path,state)
            if streak>=3:
                samples=idle_gpus(p)
                if sum(x['idle'] for x in samples)>=3 and next(x for x in samples if x['gpu']==gpu)['idle']:break
                streak=0
            time.sleep(5)
        state['capacity_sample']=samples;started=time.monotonic();signal.alarm(8990);state['status']='running';state['allocation_end']=end.isoformat();write(path,state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=p['gpus'][str(gpu)],PYTHONPATH=str(ENGINE),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        logs=ROOT/'preflight-logs';logs.mkdir(exist_ok=False)
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
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=EXTRAS,required=True);parser.add_argument('--check-admission',action='store_true');args=parser.parse_args()
    if args.check_admission:
        try:admission();print('admitted')
        except AssertionError as error:print(str(error));raise SystemExit(2)
    else:main(args.gpu)
