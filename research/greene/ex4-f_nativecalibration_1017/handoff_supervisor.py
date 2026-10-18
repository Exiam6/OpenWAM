"""Move only unstarted cells, after verified completion and safe coordinator exit."""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
HERE=ROOT/'resource-handoff'
P=json.loads((ROOT/'registration.json').read_text())
ARMS={3:'raw768_native',4:'svae48_native'}

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def guard():
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(P['deadline'])
    assert not any(Path(x).exists() for x in P['pause_paths'])

def process(pid):
    root=Path('/proc',str(pid))
    if not root.exists():return None
    try:
        fields=(root/'stat').read_text().split()
        if fields[2]=='Z':return None
        return dict(uid=root.stat().st_uid,start=fields[21],command=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None

def main():
    status=HERE/'status.json';assert not status.exists()
    frozen=json.loads((HERE/'freeze.json').read_text())
    for file,want in frozen['files'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==want,file
    stopped={};jobs=[];children={};phase='waiting_for_completed_seed42'
    identities={}
    for gpu,variant in ARMS.items():
        row=json.loads((ROOT/f'lane-gpu{gpu}.json').read_text());identities[gpu]=process(row['pid'])
        assert identities[gpu] is not None and identities[gpu]['uid']==os.getuid()
        assert str(ROOT/'code/run_lane.py') in identities[gpu]['command']
        assert f'--gpu {gpu}' in identities[gpu]['command'] and variant in identities[gpu]['command']
    def save():write(status,dict(pid=os.getpid(),phase=phase,time=dt.datetime.now(dt.timezone.utc).isoformat(),stopped=stopped,jobs=jobs))
    save()
    try:
        while len(stopped)<2:
            guard()
            for gpu,variant in ARMS.items():
                if str(gpu) in stopped:continue
                row=json.loads((ROOT/f'lane-gpu{gpu}.json').read_text());pid=row['pid'];identity=process(pid)
                assert identity==identities[gpu],f'Original coordinator is missing or changed: {gpu}'
                assert row['phase'] not in ['stopped','complete'],f'Original lane terminal before handoff: {row}'
                if row['phase']!='waiting_capacity' or row.get('seed')!=43:continue
                if row.get('admitted_samples',99)>1:continue
                records=row['records'];assert len(records)==1 and records[0].get('complete')
                assert records[0]['key']==f'{variant}-seed42'
                assert 'active_child' not in records[0]
                assert process(records[0]['server_pid']) is None
                assert all(not (ROOT/'evaluation'/f'{variant}-seed{seed}').exists() for seed in [43,44])
                assert json.loads((ROOT/'evaluation'/f'{variant}-seed42'/'L2-gate.json').read_text())['passed']
                os.kill(pid,signal.SIGTERM)
                begin=time.monotonic()
                while process(pid) is not None:
                    assert time.monotonic()-begin<30,'Coordinator exit timeout';time.sleep(.2)
                final=json.loads((ROOT/f'lane-gpu{gpu}.json').read_text())
                assert final['phase']=='stopped' and len(final['records'])==2
                assert final['records'][0]['complete'] and 'Lane terminated' in final['records'][1]['error']
                assert all(not (ROOT/'evaluation'/f'{variant}-seed{seed}').exists() for seed in [43,44])
                allocation=math.floor((10800-final['active_seconds'])/2)
                assert allocation>0
                stopped[str(gpu)]=dict(variant=variant,old_pid=pid,identity=identity,final_record=final,
                    reason='Completed seed42; terminate coordinator between cells to reschedule unstarted seed43/44.',allocated_seconds_per_remaining_cell=allocation)
                save()
            time.sleep(.5)
        phase='launching_remaining_cells';save()
        assignment=[(1,'raw768_native',43),(2,'svae48_native',43),(3,'raw768_native',44),(4,'svae48_native',44)]
        for gpu,variant,seed in assignment:
            guard();assert not (ROOT/'evaluation'/f'{variant}-seed{seed}').exists()
            budget=next(x['allocated_seconds_per_remaining_cell'] for x in stopped.values() if x['variant']==variant)
            command=['/usr/bin/python3','-u',str(HERE/'run_cell.py'),'--gpu',str(gpu),'--variant',variant,'--seed',str(seed),'--allocated-seconds',str(budget)]
            with (HERE/f'{variant}-seed{seed}.log').open('x') as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children[child.pid]=child
            jobs.append(dict(gpu=gpu,variant=variant,seed=seed,pid=child.pid,command=command,allocated_seconds=budget))
            save()
        phase='remaining_cells_running';save()
        while True:
            guard();finished=0
            for job in jobs:
                child=children[job['pid']]
                child.poll()
                path=ROOT/f"lane-gpu{job['gpu']}-{job['variant']}-seed{job['seed']}.json"
                if not path.exists():
                    assert process(job['pid']) is not None,'New worker exited before state creation';continue
                row=json.loads(path.read_text());job['phase']=row['phase']
                if row['phase']=='complete':
                    assert child.wait(timeout=30)==0,f'Worker reported complete but exited unsuccessfully: {path}'
                    finished+=1
                elif row['phase']=='stopped':raise RuntimeError(f'Remaining cell stopped: {path}')
                else:assert process(job['pid']) is not None,f'Worker disappeared: {path}'
            save()
            if finished==4:phase='complete';save();break
            time.sleep(10)
    except BaseException as exc:
        phase='supervisor_stopped';jobs.append(dict(error=repr(exc)));save();traceback.print_exc();raise

if __name__=='__main__':main()
