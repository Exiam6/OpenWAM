"""Bounded task-level collection, gated by a successful first-five feasibility pilot."""
import argparse,datetime as dt,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def admission(task):
    p=json.loads((ROOT/'protocol.json').read_text());assert task in p['tasks']
    assert not (ROOT/'tasks'/task).exists();assert not (ROOT/'paused.json').exists()
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
    assert shutil.disk_usage(ROOT).free>80*(1<<30)
    for path,digest in p['input_sha256'].items():assert sha(path)==digest,path
    return p
def main(task):
    p=admission(task);out=ROOT/'tasks'/task;out.mkdir(parents=True,exist_ok=False)
    state=dict(task=task,status='running',started=dt.datetime.now(dt.timezone.utc).isoformat(),records=[])
    child=None;start=time.monotonic()
    def terminate(*_):raise RuntimeError('Resource interruption; retain every attempt')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    env=json.loads((ROOT/'base-environment.json').read_text());gpu=os.environ['CUDA_VISIBLE_DEVICES']
    pci=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip()
    env.update(CUDA_VISIBLE_DEVICES=gpu,EXPECTED_RENDER_PCI=pci,ROBOT_CONFIG_ROOT=str(ROOT/'robot-configs'),
               ROBOTWIN_RUNTIME_ROOT=str(out/'simulator-runtime'),PYTHONDONTWRITEBYTECODE='1',
               PATH='/home/zifanz4/openwam-runtime/benchmark-env/bin:/home/zifanz4/openwam-runtime/cuda-toolkit/bin:/usr/bin:/bin')
    env.pop('PYTHONPATH',None)
    write(out/'result.json',state)
    try:
        for i,episode in enumerate(p['tasks'][task]):
            if i==5:assert any(r['status']=='complete' for r in state['records']),'L1 failed: no valid trajectory in fixed first five candidates'
            assert time.monotonic()-start<p['max_seconds']-240
            assert not (ROOT/'paused.json').exists() and shutil.disk_usage(ROOT).free>60*(1<<30)
            dest=out/f"seed-{episode['seed']}"
            with (out/f"seed-{episode['seed']}.log").open('x') as log:
                child=subprocess.Popen(['/home/zifanz4/openwam-runtime/benchmark-env/bin/python','-u',str(ROOT/'collect_one.py'),
                    '--task',task,'--seed',str(episode['seed']),'--split',episode['split'],'--out',str(dest)],
                    env=env,cwd=p['engine'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:child.wait(timeout=220)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
            row=json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else dict(episode,status='failed',error='Child exited before status')
            if row['status']=='started':row=dict(row,status='failed',error='Child stopped before completion; original status retained')
            state['records'].append(row);state['valid']=sum(r['status']=='complete' for r in state['records']);write(out/'result.json',state)
            child=None
        state.update(status='complete',collection_finished=True,training_ready=False,
                     interpretation='Dataset feasibility pilot; expert success is not policy success. No training launched by this job.')
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
            except ProcessLookupError:pass
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(finished=dt.datetime.now(dt.timezone.utc).isoformat(),occupied_seconds=time.monotonic()-start);write(out/'result.json',state)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--check-admission',action='store_true');a=p.parse_args()
    if a.check_admission:admission(a.task);print('admitted')
    else:main(a.task)
