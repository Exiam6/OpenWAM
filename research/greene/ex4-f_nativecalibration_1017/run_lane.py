"""One GPU lane, fixed checkpoint order, strict gates, no failed-attempt retries."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
R=ROOT.parent
P=json.loads((ROOT/'registration.json').read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def now():return dt.datetime.now(dt.timezone.utc).isoformat()

def guard():
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(P['deadline']),'Deadline'
    assert not any(Path(p).exists() for p in P['pause_paths']),'Pause'

def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def capacity(gpu):
    line=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    uuid,pci,free,util=[x.strip() for x in line.strip().split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
    for line in apps.splitlines():
        u,pid=[x.strip() for x in line.split(',')]
        try:
            if u==uuid and Path('/proc',pid).stat().st_uid!=os.getuid():return None
        except FileNotFoundError:pass
    return (uuid,pci) if int(free)>=32768 and int(util)<=85 else None

def main(gpu,variant):
    assert gpu in [3,4] and variant in P['variants']
    records=[];server=None;child=None;phase='waiting_capacity';active=0.;cell_start=None
    state=ROOT/f'lane-gpu{gpu}.json'
    assert not state.exists(),'Lane cannot overwrite a prior attempt'
    def save(**extra):write(state,dict(pid=os.getpid(),gpu=gpu,variant=variant,phase=phase,records=records,time=now(),**extra))
    def budget():
        guard();assert active+(time.monotonic()-cell_start if cell_start else 0)<P['lane_active_seconds'],'Lane GPU-hour budget'
    def terminated(*_):raise RuntimeError('Lane terminated')
    signal.signal(signal.SIGTERM,terminated);signal.signal(signal.SIGINT,terminated)
    lock=(R/f'gpu-{gpu}-ex4f.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for seed in P['seeds']:
            samples=[]
            while len(samples)<3:
                guard();value=capacity(gpu)
                samples=samples+[value] if value else []
                phase='waiting_capacity';save(seed=seed,admitted_samples=len(samples));time.sleep(5)
            uuid,pci=samples[-1]
            for file,want in P['files'].items():assert sha(file)==want,file
            key=f'{variant}-seed{seed}';checkpoint=P['checkpoints'][key]
            assert sha(checkpoint['path'])==checkpoint['sha256']
            cell=ROOT/'evaluation'/key;cell.mkdir(parents=True,exist_ok=False)
            cell_start=time.monotonic();record=dict(key=key,started=now(),groups=[]);records.append(record)
            env=json.loads(Path(P['environment_source']).read_text())
            for key_to_drop in ['REPLAY_HELPERS','REPLAY_PROPRIO','REPLAY_FRAMES']:env.pop(key_to_drop,None)
            env.update(CUDA_VISIBLE_DEVICES=uuid,EXPECTED_RENDER_PCI=pci,
                PYTHONPATH='/home/zifanz4/openwam-rae-research',ROBOT_CONFIG_ROOT=str(ROOT/'robot-configs'),
                PATH=f'{R}/policy-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin')
            port=19940+gpu
            with socket.socket() as s:s.bind(('127.0.0.1',port))
            write(cell/'environment.json',env)
            def launch(command,runenv,logfile):
                with logfile.open('x') as log:
                    return subprocess.Popen(command,env=runenv,cwd='/home/zifanz4/openwam-rae-research',stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            servercmd=[str(R/'policy-env/bin/python'),'-u',str(ROOT/'fixed/serve_policy.py'),'--checkpoint',checkpoint['path'],'--port',str(port),'--log',str(cell/'episode-rng.jsonl')]
            server=launch(servercmd,env,cell/'server.log');record.update(server_pid=server.pid,server_command=servercmd)
            phase='loading_policy';save();start=time.monotonic()
            while 'ENDTOEND_SERVER_READY' not in (cell/'server.log').read_text(errors='replace'):
                budget();assert server.poll() is None,'Policy server failed';assert time.monotonic()-start<900;time.sleep(3)
            def simulation(condition,command,logfile,timeout):
                nonlocal child
                simenv=dict(env,PATH=f'{R}/benchmark-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin',ROBOTWIN_RUNTIME_ROOT=str(cell/'simulator-runtime'/condition))
                simenv.pop('PYTHONPATH',None)
                child=launch(command,simenv,logfile);record['active_child']=dict(pid=child.pid,command=command,condition=condition);save()
                start=time.monotonic()
                while child.poll() is None:
                    budget();assert server.poll() is None,'Server exited';assert time.monotonic()-start<timeout,'Simulation timeout';time.sleep(3)
                assert child.returncode==0,f'Simulator exited {child.returncode}: {logfile}'
                child=None;record.pop('active_child',None)
            for condition in ['old','fixed']:
                phase='training_scene_smoke';dest=cell/f'smoke-{condition}';dest.mkdir()
                simulation(condition,[str(R/'benchmark-env/bin/python'),'-u',str(ROOT/condition/'simulation_smoke.py'),'--port',str(port),'--output',str(dest)],dest/'run.log',600)
                assert json.loads((dest/'result.json').read_text())['passed']
                write(ROOT/condition/f'simulation-preflight-{variant}-{seed}.json',dict(passed=True,checkpoint=checkpoint,
                    code_sha256={f:h for f,h in P['files'].items() if str(ROOT/condition) in f},result=str(dest/'result.json')))
            groups=[('fixed','adjust_bottle',P['selected']['adjust_bottle']),
                ('old','handover_block',P['selected']['handover_block'][:1]),
                ('fixed','handover_block',P['selected']['handover_block']),
                ('fixed','place_object_basket',P['selected']['place_object_basket'])]
            for condition,task,seeds in groups:
                phase='closed_loop';dest=cell/condition/task;dest.mkdir(parents=True)
                command=[str(R/'benchmark-env/bin/python'),'-u',str(ROOT/condition/'evaluate_policy.py'),
                    '--scene-seeds',json.dumps(seeds),'--task',task,'--route',variant,'--seed',str(seed),
                    '--condition','clean','--port',str(port),'--output',str(dest)]
                simulation(condition,command,dest/'run.log',len(seeds)*600+180)
                result=json.loads((dest/'summary.json').read_text());record['groups'].append(dict(condition=condition,task=task,summary=str(dest/'summary.json'),successes=result['successes'],episodes=result['episodes']));save()
                assert result['complete'] and result['technical_failures']==0,result
                if condition=='old':assert result['successes']==0,'Old path does not reproduce baseline'
                if task=='adjust_bottle' and seed==42:
                    assert result['successes']>=1,'L2 easy-task success not reproduced'
                    write(cell/'L2-gate.json',dict(passed=True,result=str(dest/'summary.json')))
            stop(server);server=None;active+=time.monotonic()-cell_start;cell_start=None
            record.update(complete=True,finished=now());write(cell/'result.json',record);save(active_seconds=active)
        phase='complete'
    except BaseException as exc:
        phase='stopped';records.append(dict(error=repr(exc),time=now()));traceback.print_exc();raise
    finally:
        stop(child);stop(server);save(active_seconds=active+(time.monotonic()-cell_start if cell_start else 0));lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',required=True);a=p.parse_args();main(a.gpu,a.variant)
