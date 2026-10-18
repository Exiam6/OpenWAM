"""CPU-only evidence queue: encode a task/seed only after both cells complete."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
CODE=Path(__file__).resolve().parent

def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    p=json.loads((ROOT/'registration.json').read_text());state=ROOT/'media-queue.json';assert not state.exists()
    jobs=[(task,seed) for task in p['selected'] for seed in p['seeds']];done={};active={};logs=ROOT/'media-queue-logs';logs.mkdir(exist_ok=False)
    def save(status,**extra):write(state,dict(pid=os.getpid(),time=stamp(),status=status,complete=len(done),planned=len(jobs),done=list(done.values()),active=[dict(task=k[0],seed=k[1],pid=v['process'].pid) for k,v in active.items()],**extra))
    def stop(*_):raise RuntimeError('Media queue stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save('starting')
    try:
        while len(done)<len(jobs):
            assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,18,17,tzinfo=dt.timezone.utc),'Media queue deadline'
            assert not any(Path(x).exists() for x in p['pause_paths']),'Pause requested'
            for key in list(active):
                child=active[key]['process']
                if child.poll() is None:continue
                assert child.returncode==0,('Encoder failed; preserve partial output',key,child.returncode)
                del active[key]
            for task,seed in jobs:
                key=(task,seed)
                if key in done or key in active:continue
                manifest=ROOT/f'paired-videos-{task}-seed{seed}.json'
                if manifest.exists():
                    d=json.loads(manifest.read_text());rows=d['records']
                    assert len(rows)==10 and len({(x['cell'],x['scene_seed']) for x in rows})==10
                    assert all(x['full_decode_passed'] for x in rows)
                    done[key]=dict(task=task,seed=seed,manifest=str(manifest),pairs=len(rows));continue
                ready=True
                for variant in p['variants']:
                    path=ROOT/'evaluation'/f'{variant}-seed{seed}'/'fixed'/task/'summary.json'
                    if not path.exists():ready=False;break
                    d=json.loads(path.read_text());assert d['complete'] and d['episodes']==5 and d['technical_failures']==0
                if not ready or len(active)>=2:continue
                command=[str(ROOT.parent/'policy-env/bin/python'),'-u',str(CODE/'make_pairs.py'),'--task',task,'--seed',str(seed)]
                with (logs/f'{task}-seed{seed}.log').open('x') as log:
                    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
                active[key]=dict(process=child,command=command)
            save('encoding_or_waiting')
            if len(done)<len(jobs):time.sleep(10)
        save('complete')
    except BaseException as exc:
        save('failed',error=repr(exc));raise
    finally:
        for row in active.values():
            child=row['process']
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

if __name__=='__main__':main()
