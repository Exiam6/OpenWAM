"""Bound an explicitly gated training or evaluation batch on dedicated idle GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from build_cache import sha
from stage_runner import run_child

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018');SOURCE=Path(__file__).parent
ADMISSION_POLL_SECONDS=5.


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def admission(phase,seeds):
    path=ROOT/'gpu-registration.json';assert path.exists(),'CPU-only: GPU stage not registered'
    p=json.loads(path.read_text());assert p['gpu_launch_permitted']
    assert seeds and len(seeds)==len(set(seeds)) and set(seeds)<={42,43,44}
    for file,digest in p['batch_files'].items():assert sha(file)==digest,file
    assert str(Path(__file__).resolve()) in p['batch_files'] and str(SOURCE/'stage_runner.py') in p['batch_files']
    if phase=='training':
        for file,digest in p['asset_files'].items():assert sha(file)==digest,file
        assert set(seeds)=={42,43,44},'Allthree registered training seeds required'
        from train_job import admission as check
        check('full')
        limits={seed:10800. for seed in seeds}
    else:
        from evaluate_job import admission as check
        stage='pilot' if phase=='pilot_evaluation' else 'full'
        # Fixed endpoints advance independently; final study still requires allthree registered seeds.
        for seed in seeds:
            trained=json.loads((ROOT/'runs'/f'vae48_native-seed{seed}'/'result.json').read_text())
            assert trained['complete'] and trained['steps']==20000
        limits={seed:float(check(seed,stage)[2]) for seed in seeds}
    for seed in seeds:
        destination=ROOT/'runs'/f'vae48_native-seed{seed}' if phase=='training' else ROOT/'evaluation'/f'seed{seed}'/('pilot' if phase=='pilot_evaluation' else 'full')
        assert not destination.exists(),'Previous scientific attempt must be preserved: '+str(destination)
    return p,limits


def allocation_end(p,gpu,required):
    now=dt.datetime.now(dt.timezone.utc);end=dt.datetime.fromisoformat(p['deadline'])-dt.timedelta(seconds=60)
    if gpu not in [1,2,3,4]:
        assert dt.datetime.fromisoformat(p['extra_window_start'])<=now
        end=min(end,dt.datetime.fromisoformat(p['extra_window_end'])-dt.timedelta(seconds=60))
    assert (end-now).total_seconds()>required+60,'Complete remaining endpoint does not fit allocation'
    return end


def capacity(p,gpus):
    occupied={line.split(',')[0].strip() for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines() if line.strip()}
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    samples={}
    for line in output.splitlines():
        index,uuid,pci,free,total,util=[x.strip() for x in line.split(',')];gpu=int(index)
        if gpu not in gpus:continue
        assert uuid==p['gpus'][str(gpu)]
        idle=uuid not in occupied and int(util)==0 and int(total)-int(free)<256
        samples[gpu]=dict(gpu=gpu,uuid=uuid,pci=pci,free_mib=int(free),util=int(util),idle=idle)
    assert set(samples)==set(gpus),samples
    return samples


def main(phase,seeds,gpus):
    p,limits=admission(phase,seeds);assert len(gpus)==len(seeds) and len(set(gpus))==len(gpus)
    name=phase+'-'+'-'.join(map(str,seeds));root=ROOT/'batches'/name;root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();state=dict(phase=phase,pid=os.getpid(),status='capacity_check',seeds=seeds,gpus=gpus,started=dt.datetime.now(dt.timezone.utc).isoformat(),children=[])
    locks=[];executor=None;cancel=threading.Event()
    def save():write(root/'result.json',state)
    def stop(*_):raise RuntimeError('Batch supervisor termination')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);save()
    try:
        for gpu in sorted(gpus):
            lock=(ROOT/f'gpu-{gpu}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append((gpu,lock))
        ends={};streak=0
        while True:
            for seed,gpu in zip(seeds,gpus):
                required=limits[seed]+11700 if phase=='training' else (11700 if phase=='pilot_evaluation' else limits[seed])
                ends[seed]=allocation_end(p,gpu,required)
            samples=capacity(p,gpus);streak=streak+1 if all(x['idle'] for x in samples.values()) else 0
            state.update(capacity_samples=samples,admitted_samples=streak,status='waiting_capacity');save()
            if streak>=3:
                samples=capacity(p,gpus)
                if all(x['idle'] for x in samples.values()):break
                streak=0
            time.sleep(ADMISSION_POLL_SECONDS)
        claims=ROOT/'phase-claims';claims.mkdir(exist_ok=True)
        for seed in seeds:
            with (claims/f'{phase}-seed{seed}.json').open('x') as f:
                json.dump(dict(status='reserved',supervisor_pid=os.getpid(),batch=str(root),seed=seed),f)
        state['status']='running';save();executor=ThreadPoolExecutor(max_workers=len(seeds))
        def execute(seed,gpu):
            directory=root/f'seed{seed}';directory.mkdir();claim=directory/'execution.json'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=samples[gpu]['uuid'],EXPECTED_RENDER_PCI=samples[gpu]['pci'],PYTHONPATH='/home/zifanz4/openwam-rae-research',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            if phase=='training':
                command=[sys.executable,'-u',str(SOURCE/'train_job.py'),'--seed',str(seed),'--stage','full']
                result_path=ROOT/'runs'/f'vae48_native-seed{seed}'/'result.json'
            else:
                stage='pilot' if phase=='pilot_evaluation' else 'full'
                command=[sys.executable,'-u',str(SOURCE/'evaluate_job.py'),'--seed',str(seed),'--stage',stage,'--gpu',str(gpu),'--allocation-end',ends[seed].isoformat()]
                result_path=ROOT/'evaluation'/f'seed{seed}'/stage/'result.json'
            def record_pid(pid):
                value=dict(status='running',pid=pid,seed=seed,gpu=gpu,command=command,allocation_end=ends[seed].isoformat())
                write(claim,value);write(claims/f'{phase}-seed{seed}.json',value)
            result=run_child(command,log=directory/'run.log',seconds=limits[seed],deadline=ends[seed],pause_paths=p['pause_paths'],env=env,cwd='/home/zifanz4/openwam-rae-research',
                health_check=lambda:'Batch termination requested' if cancel.is_set() else None,on_started=record_pid)
            result.update(seed=seed,gpu=gpu,result_path=str(result_path))
            if result['complete']:
                try:
                    evidence=json.loads(result_path.read_text());result['complete']=bool(evidence.get('complete',False));result['result_sha256']=sha(result_path)
                except Exception as error:result.update(complete=False,error='Missing or invalid child evidence: '+repr(error))
            write(claim,result);write(claims/f'{phase}-seed{seed}.json',result)
            return result
        futures=[executor.submit(execute,seed,gpu) for seed,gpu in zip(seeds,gpus)]
        for future in as_completed(futures):
            result=future.result();state['children'].append(result);save()
            next(lock for gpu,lock in locks if gpu==result['gpu']).close()
        state.update(status='complete' if all(x['complete'] for x in state['children']) else 'failed',complete=all(x['complete'] for x in state['children']),
            occupied_gpu_seconds=sum(x['occupied_seconds'] for x in state['children']),next_stage_automatically_released=False)
    except BaseException as error:
        cancel.set();state.update(status='failed',complete=False,error=repr(error));raise
    finally:
        if executor is not None:executor.shutdown(wait=True)
        for _,lock in locks:lock.close()
        state['wall_seconds']=time.monotonic()-started;state['finished']=dt.datetime.now(dt.timezone.utc).isoformat();save()
    print(json.dumps(state))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['training','pilot_evaluation','full_evaluation'],required=True);parser.add_argument('--seeds',nargs='+',type=int,required=True);parser.add_argument('--gpus',nargs='+',type=int,choices=range(8));parser.add_argument('--check-admission',action='store_true');args=parser.parse_args()
    if args.check_admission:
        try:admission(args.phase,args.seeds);print('admitted')
        except AssertionError as error:print(str(error));raise SystemExit(2)
    else:
        assert args.gpus;main(args.phase,args.seeds,args.gpus)
