"""Observe owned jobs, retain half-hour GPU audits, and finalize once complete."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_cache import sha
from full_media import ROOT, write

RUN = ROOT / 'resumption-20261019-v1'
SOURCE = Path(__file__).parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def live(record):
    path = Path('/proc') / str(record['pid'])
    try:
        state = (path / 'stat').read_text().rsplit(')',1)[1].split()[0]
        command = (path / 'cmdline').read_bytes().split(b'\0')
        return state != 'Z' and str(record['command'][2]).encode() in command
    except FileNotFoundError:
        return False


def main():
    auth = read(RUN / 'authorization.json')
    registration = read(ROOT / 'gpu-registration.json')
    deadline = dt.datetime.fromisoformat(auth['deadline'])
    batch_launch, media_launch = (read(RUN / name) for name in ['batch-process.json','media-process.json'])
    assert batch_launch and media_launch
    manifest = read(ROOT / 'resumption-evidence-code-v1-manifest.json')
    for path, digest in manifest['files'].items(): assert sha(path) == digest, path
    assert manifest['authorization_sha256'] == sha(RUN / 'authorization.json')
    state = dict(pid=os.getpid(), status='observing', started=dt.datetime.now(dt.timezone.utc).isoformat(),
                 audit_interval_seconds=1800, poll_seconds=30, automatic_gpu_retry=False)
    next_audit = 0
    try:
        while True:
            now = dt.datetime.now(dt.timezone.utc)
            assert now < deadline, 'Authorized continuation window expired'
            assert not any(Path(p).exists() for p in registration['pause_paths']), 'Pause requested'
            batch = read(ROOT / 'batches/full_evaluation-42-43/result.json')
            media = read(ROOT / 'full-media-continuation-v1/queue.json')
            batch_alive, media_alive = live(batch_launch), live(media_launch)
            state.update(time=now.isoformat(), batch_alive=batch_alive, media_alive=media_alive,
                         batch_status=batch.get('status') if batch else None,
                         media_status=media.get('status') if media else None,
                         completed_pairs=media.get('completed') if media else None)
            write(RUN / 'watch-state.json',state)
            if time.monotonic() >= next_audit:
                gpu = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv'],
                                     capture_output=True,text=True,timeout=20)
                applications = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],
                                              capture_output=True,text=True,timeout=20)
                audit = dict(state, gpu_returncode=gpu.returncode, gpu=gpu.stdout, gpu_error=gpu.stderr,
                             applications_returncode=applications.returncode, applications=applications.stdout)
                with (RUN / 'resource-audits.jsonl').open('a') as handle: handle.write(json.dumps(audit)+'\n')
                next_audit = time.monotonic()+1800
            assert not batch or batch['status'] != 'failed', 'Evaluation batch failed; preserve attempt'
            assert not media or media['status'] != 'failed', 'Media continuation failed; preserve attempt'
            assert batch_alive or (batch and batch.get('complete')), 'Evaluation parent exited without completion'
            assert media_alive or (media and media['status']=='complete'), 'Media queue exited without completion'
            if batch and batch.get('complete') and media and media['status']=='complete' and not batch_alive and not media_alive:
                break
            time.sleep(30)
        state.update(status='generating_complete_report')
        write(RUN / 'watch-state.json',state)
        command = [sys.executable,'-u',str(SOURCE / 'report_complete.py')]
        with (RUN / 'report.log').open('x') as log:
            report = subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                                    timeout=max(1,(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()))
        assert report.returncode == 0, 'Report generation failed; preserve artifacts'
        complete = read(ROOT / 'complete-evidence.json')
        assert complete['complete'] and complete['paired_videos']==360 and complete['evaluations_per_arm']==180
        assert sha(complete['bundle'])==complete['sha256']
        state.update(status='complete',complete_evidence=str(ROOT / 'complete-evidence.json'),bundle_sha256=complete['sha256'])
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        state['finished']=dt.datetime.now(dt.timezone.utc).isoformat()
        write(RUN / 'watch-state.json',state)


if __name__=='__main__':
    main()
