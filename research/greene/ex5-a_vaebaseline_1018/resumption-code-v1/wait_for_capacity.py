"""One CPU-only admission wait for all three corrected full-training seeds."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from corrected_protocol import ROOT,SOURCE,ENGINE,load_registration,verify_training_release,idle_capacity,sha

REQUIRED=10800+11700+60
POLL_SECONDS=30


def eligible_gpus(p,samples,current):
    eligible=[]
    for gpu,sample in sorted(samples.items()):
        if not sample['idle']:continue
        end=dt.datetime.fromisoformat(p['deadline'])-dt.timedelta(seconds=60)
        if gpu not in [1,2,3,4]:
            if current<dt.datetime.fromisoformat(p['extra_window_start']):continue
            end=min(end,dt.datetime.fromisoformat(p['extra_window_end'])-dt.timedelta(seconds=60))
        if (end-current).total_seconds()>REQUIRED:eligible.append(gpu)
    return eligible


def main():
    p=load_registration();verify_training_release()
    target=ROOT/'corrected-training-capacity-wait-v1.json'
    lock=(ROOT/'corrected-training-capacity-wait-v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not target.exists(),'Preserve any earlier wait attempt; no automatic restart'
    assert not (ROOT/'batches/training-42-43-44').exists()
    state=dict(pid=os.getpid(),status='waiting_capacity',started=dt.datetime.now(dt.timezone.utc).isoformat(),protocol_sha256=sha(SOURCE/'protocol.json'),seeds=[42,43,44],required_remaining_seconds=REQUIRED,scientific_gpu_attempt_started=False)
    def save():
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(target)
    def stop(*_):raise RuntimeError('Capacity wait terminated')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    child=None;save()
    try:
        while True:
            current=dt.datetime.now(dt.timezone.utc)
            assert (dt.datetime.fromisoformat(p['deadline'])-dt.timedelta(seconds=60)-current).total_seconds()>REQUIRED,'Full endpoint admission window expired before launch'
            for pause in p['pause_paths']:assert not Path(pause).exists()
            samples=idle_capacity(p,list(range(8)));eligible=eligible_gpus(p,samples,current)
            state.update(time=current.isoformat(),capacity_samples=samples,eligible_gpus=eligible);save()
            if len(eligible)>=3:
                chosen=eligible[:3]
                # The batch itself rechecks source/assets, GPU locks, three idle polls,
                # fresh final availability and the entire remaining endpoint window.
                command=[sys.executable,'-u',str(SOURCE/'run_batch.py'),'--phase','training','--seeds','42','43','44','--gpus',*map(str,chosen)]
                env=dict(os.environ,PYTHONPATH=str(ENGINE),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                with (ROOT/'corrected-training-batch-v1.log').open('x') as log:
                    child=subprocess.Popen(command,env=env,cwd=ENGINE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='batch_running',batch_pid=child.pid,selected_gpus=chosen,command=command);save()
                while child.poll() is None:time.sleep(1)
                result=ROOT/'batches/training-42-43-44/result.json'
                evidence=json.loads(result.read_text()) if result.exists() else {}
                state.update(status='complete' if child.returncode==0 and evidence.get('complete') else 'failed',batch_exit_code=child.returncode,batch_result=str(result),batch_result_sha256=sha(result) if result.exists() else None,
                    scientific_gpu_attempt_started=any((ROOT/'runs'/f'vae48_native-seed{s}').exists() for s in [42,43,44]),next_stage_automatically_released=False)
                break
            time.sleep(POLL_SECONDS)
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        state['finished']=dt.datetime.now(dt.timezone.utc).isoformat();save();lock.close()


if __name__=='__main__':main()
