"""One bounded registered experiment, confined to one idle GPU and its own process group."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from policy import validate_job, window
from probe import probe


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write(path, data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def owned_process(pid, child_pid):
    """Policy servers may create their own session while remaining in our systemd unit."""
    try:
        own=(Path('/proc')/str(os.getpid())/'cgroup').read_text().splitlines()
        other=(Path('/proc')/str(pid)/'cgroup').read_text().splitlines()
        unit_paths=[line.split(':',2)[2] for line in own if 'openwam-nightly-job-' in line]
        if unit_paths:
            return any(line.split(':',2)[2]==unit or line.split(':',2)[2].startswith(unit+'/')
                       for line in other for unit in unit_paths)
        current=pid
        for _ in range(128):
            if current in [child_pid,os.getpid()]:return True
            if current<=1:return False
            current=int((Path('/proc')/str(current)/'stat').read_text().rsplit(')',1)[1].split()[1])
        return False
    except FileNotFoundError:
        return True  # Process already exited and no longer occupies the GPU.
    except PermissionError:
        return False


def run(spec_path, host, gpu_uuid, end_string, runtime):
    started=time.monotonic();job=validate_job(json.loads(spec_path.read_text()))
    out=runtime/'runs'/job['id'];out.mkdir(parents=True,exist_ok=False)
    state=dict(id=job['id'],scientific_identity=job['scientific_identity'],status='validating',pid=os.getpid(),
               host=host,gpu_uuid=gpu_uuid,started=dt.datetime.now(dt.timezone.utc).isoformat(),spec_sha256=sha(spec_path))
    child=None
    def save():state['occupied_parent_seconds']=time.monotonic()-started;write(out/'result.json',state)
    def stop_signal(*_):raise RuntimeError('Worker terminated')
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    signal.signal(signal.SIGALRM,stop_signal);signal.alarm(max(1,job['max_seconds']-30));save()
    try:
        manifest=json.loads((runtime/'source-manifest.json').read_text())
        assert manifest['files'][str(Path(__file__).resolve())]==sha(Path(__file__).resolve())
        for path,digest in manifest['files'].items():assert sha(path)==digest,path
        end=dt.datetime.fromisoformat(end_string)
        now=dt.datetime.now(dt.timezone.utc);_,window_end,inside=window(now)
        assert inside and end<=window_end-dt.timedelta(seconds=60)
        assert (end-now).total_seconds()>job['max_seconds']+30
        assert sha(job['authorization_evidence'])==job['authorization_sha256']
        command=job['host_commands'][host]
        assert not any(Path(p).exists() for p in job.get('pause_paths',[]))
        assert not any(Path(p).exists() for p in command['outputs_absent'])
        for path,digest in command['input_sha256'].items():assert sha(path)==digest,path
        locks=Path('/home/zifanz4/.local/state/openwam-gpu-locks');locks.mkdir(parents=True,exist_ok=True)
        with (locks/f'{gpu_uuid}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (out/'admission.log').open('x') as log:
                check=subprocess.run(command['admission_argv'],env=env,cwd=command['cwd'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=60)
            assert check.returncode==0,'Registered experiment admission refused'
            row=next(r for r in probe()['gpus'] if r['uuid']==gpu_uuid)
            assert row['idle'],'GPU occupied before launch'
            assert not any(Path(p).exists() for p in command['outputs_absent'])
            assert time.monotonic()-started<job['max_seconds']-60,'Insufficient budget after admission'
            env.update(CUDA_VISIBLE_DEVICES=gpu_uuid,OPENWAM_ALLOCATION_END=end.isoformat(),
                       OPENWAM_GPU_INDEX=str(row['index']),OPENWAM_GPU_HOST=host)
            with (out/'run.log').open('x') as log:
                child=subprocess.Popen(command['argv'],env=env,cwd=command['cwd'],stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',child_pid=child.pid,command=command['argv'],gpu_index=row['index']);save()
            while child.poll() is None:
                assert time.monotonic()-started<job['max_seconds']-45,'Registered job cap reached'
                assert dt.datetime.now(dt.timezone.utc)<end-dt.timedelta(seconds=45),'Nightly deadline reached'
                assert not any(Path(p).exists() for p in job.get('pause_paths',[]))
                current=next(r for r in probe()['gpus'] if r['uuid']==gpu_uuid)
                assert all(owned_process(p['pid'],child.pid) for p in current['processes']),'Another process started using this GPU; stop only our job'
                save();time.sleep(5)
            state.update(status='complete' if child.returncode==0 else 'failed',exit_code=child.returncode)
    except BaseException as error:
        state.update(status='failed',error=repr(error))
    finally:
        signal.alarm(0)
        if child is not None:
            # Also clean up descendants after an early parent exit. Never signal another process group.
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=5)
        state['finished']=dt.datetime.now(dt.timezone.utc).isoformat();save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--host',required=True);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--allocation-end',required=True);parser.add_argument('--runtime',type=Path,required=True)
    args=parser.parse_args();run(args.spec,args.host,args.gpu_uuid,args.allocation_end,args.runtime)
