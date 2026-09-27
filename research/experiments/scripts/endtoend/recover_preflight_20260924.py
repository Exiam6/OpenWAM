"""One-off, approval-gated recovery of two verified zero-step launch failures.

Default mode is read-only. Frozen experiment/job/controller code is not edited.
Original failure files are moved into an immutable attempt archive before a retry.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import shlex
import socket
import subprocess
import time
from pathlib import Path

P = Path('/home/zifanz4/openwam-experiments')
S = P / 'studies/endtoend-20260923'
E = Path('/data02/zifanz4/openwam-experiments/endtoend-20260923')
PLAN = S / 'preflight-recovery-20260924T0230.json'
APPROVAL = S / 'preflight-recovery-20260924T0230-approved.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return dt.datetime.now().astimezone()


def remote(args):
    return subprocess.check_output(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
         'zifanz4@cm009.csl.illinois.edu', shlex.join(args)],
        text=True, timeout=30)


def check(plan):
    assert socket.gethostname().split('.')[0] == 'cm001'
    for path in plan['pause_paths']:
        assert not Path(path).exists(), path
    for path, digest in plan['required_sha256'].items():
        assert sha(Path(path)) == digest, path
    assert now() < dt.datetime.fromisoformat(plan['dependency_deadline'])
    assert now() < dt.datetime.fromisoformat(plan['original_controller_deadline'])
    assert not Path(plan['archive']).exists(), 'This recovery is single-use'
    assert read(E / 'pipeline/exit.json')['returncode'] == 1
    for name in plan['runs']:
        root = E / 'training' / name
        assert read(root / 'exit.json')['returncode'] == 1
        for relative in ['start.json', 'run.log', 'metrics-rank0.jsonl',
                         'batch-stream.jsonl', 'initialization.json', 'output']:
            assert not (root / relative).exists(), (name, relative)
    for path in (E / 'training').glob('*/exit.json'):
        if path.parent.name not in plan['runs']:
            assert read(path)['returncode'] == 0, ('New failure needs separate review', str(path))
    for route in ['pca', 'svae', 'wan']:
        for seed in [42, 43, 44]:
            job = read(E / 'training' / f'{route}-seed{seed}/job.json')
            for path, digest in job['code_sha256'].items():
                assert sha(Path(path)) == digest, path
    assert not (S / 'evaluation-ready.json').exists()
    assert not (S / 'fresh-cohort-integrity.json').exists()
    assert not list(E.glob('evaluation-slots/*/started.json'))
    # Keep the frozen idle/ECC/process criteria; require three clean snapshots.
    samples = []
    for sample in range(3):
        rows = remote(['nvidia-smi', '-i', ','.join(sorted(plan['gpu_uuids'])), '--query-gpu=index,uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total', '--format=csv,noheader,nounits'])
        processes = remote(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'])
        seen = set()
        for line in rows.splitlines():
            index, uuid, memory, utilization, ecc = [x.strip() for x in line.split(',')]
            assert plan['gpu_uuids'][index] == uuid
            assert int(memory) < 100 and int(utilization) == 0 and int(ecc) == 0
            assert uuid not in processes
            seen.add(index)
        assert seen == set(plan['gpu_uuids'])
        samples.append({'time': now().isoformat(), 'gpu_rows': rows, 'compute_uuids': processes})
        if sample != 2:
            time.sleep(2)
    return samples


def execute(plan, samples):
    approval = read(APPROVAL)
    assert approval['explicit_user_approval'] is True
    assert approval['plan_sha256'] == sha(PLAN)
    assert approval['script_sha256'] == sha(Path(__file__))
    assert approval['user_instruction'].strip()
    archive = Path(plan['archive'])
    archive.mkdir(parents=True, exist_ok=False)
    record = {'time': now().isoformat(), 'approval': approval, 'idle_checks': samples,
              'moves': [], 'launched': [], 'status': 'executing'}
    def save():
        (archive / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    save()
    try:
        for name in plan['runs']:
            dest = archive / name
            dest.mkdir()
            for filename in ['exit.json', 'launcher-error.json']:
                source = E / 'training' / name / filename
                digest = sha(source)
                source.rename(dest / filename)
                assert sha(dest / filename) == digest
                record['moves'].append({'from': str(source), 'to': str(dest / filename), 'sha256': digest})
                save()
        # Preserve original controller and queue failures; never overwrite them.
        (E / 'pipeline').rename(archive / 'pipeline')
        (E / 'pipeline-controller.log').rename(archive / 'pipeline-controller.log')
        for slot in [1, 2]:
            source = E / 'training' / f'queue-gpu{slot}.json'
            (archive / source.name).write_bytes(source.read_bytes())
        for name in plan['runs']:
            session = 'ow-recovery-' + name
            # Resource placement only; original jobs/configs remain unchanged.
            job = read(E / 'training' / name / 'job.json')
            slot = plan['retry_gpu_index'][name]
            job['gpu_index'] = slot
            job['gpu_uuid'] = plan['gpu_uuids'][str(slot)]
            retry_job = archive / (name + '-job.json')
            retry_job.write_text(json.dumps(job, indent=2) + '\n')
            command = shlex.join(['timeout', '--signal=TERM', '--kill-after=30s', '86430',
                                 'python3', str(P / 'scripts/endtoend/run_job.py'),
                                 str(retry_job)])
            command += ' > ' + shlex.quote(str(archive / (name + '.log'))) + ' 2>&1'
            remote(['tmux', 'new-session', '-d', '-s', session, command])
            record['launched'].append(session)
            save()
        # Do not restart the original wall-clock budget when resuming dependencies.
        remaining = math.floor((dt.datetime.fromisoformat(plan['original_controller_deadline']) - now()).total_seconds())
        assert remaining > 0
        command = shlex.join(['timeout', '--signal=TERM', '--kill-after=30s', str(remaining),
                             'python3', str(P / 'scripts/endtoend/pipeline_controller.py')])
        command += ' > ' + shlex.quote(str(E / 'pipeline-controller.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'ow-e2e-pipeline', command], check=True)
        record['launched'].append('ow-e2e-pipeline')
        record['status'] = 'launched_once_no_automatic_retry'
    except Exception as error:
        record['status'] = 'recovery_failed_no_automatic_retry'
        record['error'] = repr(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = read(PLAN)
    samples = check(plan)
    if args.execute:
        execute(plan, samples)
    print(json.dumps({'time': now().isoformat(), 'read_only': not args.execute,
                      'passed': True, 'idle_checks': samples, 'runs': plan['runs']}))
