"""One finite first-start evaluation stage; frozen experiment functions are unchanged.

No retries, no training, no scoring of old cohorts. Each original slot independently
waits for three idle samples. SIGTERM/deadline raises through the original queue's
finally blocks so only its own policy server and evaluator receive termination.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

import pipeline_common as pc

S, E, R, P = pc.S, pc.E, pc.R, pc.P
PLAN = S / 'evaluation-recovery-20260925.json'
OUT = E / 'recovery/evaluation-20260925'
SCRIPT = Path(__file__).resolve()
TASKS = ['adjust_bottle', 'handover_block', 'place_object_basket']
ROUTES = ['svae', 'pca', 'wan']


class StageStopped(BaseException):
    pass


def interrupted(signum, frame):
    raise StageStopped(f'Owned stage received signal {signum}')


def protect(seconds):
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    assert seconds > 0
    signal.setitimer(signal.ITIMER_REAL, seconds)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def loadplan():
    q = json.loads(PLAN.read_text())
    assert digest(SCRIPT) == q['recovery_script_sha256']
    assert digest(S / 'pipeline-plan.json') == q['original_plan_sha256']
    pc.frozen(json.loads((S / 'pipeline-plan.json').read_text())['code_sha256'])
    return q


def guard(q, evaluation=False):
    pc.guard()
    for p in q['pause_paths']:
        assert not Path(p).exists(), p
    key = 'evaluation_deadline' if evaluation else 'stage_deadline'
    assert pc.now() < datetime.datetime.fromisoformat(q[key]), f'Fixed {key} reached'


def remaining(q, evaluation=False):
    key = 'evaluation_deadline' if evaluation else 'stage_deadline'
    return (datetime.datetime.fromisoformat(q[key]) - pc.now()).total_seconds()


def marker(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def identity():
    return {'time': pc.now().isoformat(), 'pid': os.getpid(), 'uid': os.getuid(),
            'host': socket.gethostname(),
            'start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            'script_sha256': digest(SCRIPT), 'plan_sha256': digest(PLAN)}


def check(q):
    guard(q)
    assert json.loads((E / 'pipeline/exit.json').read_text())['returncode'] == 1
    ex = json.loads((E / 'evaluation-resource-wait-20260924/exit.json').read_text())
    assert ex['status'] == 'original_dependency_deadline_reached'
    for name in ['pipeline_controller.py', 'wait_evaluation_resources_20260924.py']:
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = p.read_bytes().split(b'\0')
                assert str(P / name).encode() not in args, ('Old stage still alive', str(p))
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
    for route in ROUTES:
        for seed in [42, 43, 44]:
            root = E / 'training' / f'{route}-seed{seed}'
            assert json.loads((root / 'exit.json').read_text())['returncode'] == 0
    for p in [E / 'collection-slots/5/exit.json', E / 'collection-slots/7/exit.json',
              E / 'simulation-preflight/exit.json']:
        assert json.loads(p.read_text())['returncode'] == 0
    assert json.loads((S / 'simulation-preflight.json').read_text())['passed']
    for p in [S / 'fresh-cohort-integrity.json', S / 'evaluation-ready.json',
              E / 'evaluation-slots', E / 'evaluation', E / 'action-diagnostics']:
        assert not p.exists(), ('Not a fresh first start', str(p))
    assert shutil.disk_usage(E).free > q['minimum_free_bytes']
    manifest = json.loads((OUT / 'previous-terminal-evidence/manifest.json').read_text())
    for x in manifest['files'].values():
        assert digest(x['archive']) == x['sha256']
    return {'passed': True, 'time': pc.now().isoformat(), 'training_exit0': 9,
            'old_stages_terminal': True, 'existing_eval_artifacts': False,
            'original_scientific_hashes_match': True, 'free_bytes': shutil.disk_usage(E).free}


def gate(q):
    check(q)
    dest = OUT / 'gate'
    marker(dest / 'started.json', identity())
    protect(min(q['gate_timeout_seconds'], remaining(q)))
    code = 1
    try:
        rc = pc.run_child([str(R / 'benchmark-env/bin/python'), str(P / 'verify_fresh_cohort.py')],
                          os.environ.copy(), dest / 'cohort.log', 1800)
        assert rc == 0, 'Fresh cohort verification failed; retain artifact, no retry'
        cp, streams = {}, {}
        freeze = dict(json.loads((S / 'pipeline-plan.json').read_text())['code_sha256'])
        freeze[str(SCRIPT)] = digest(SCRIPT)
        freeze[str(PLAN)] = digest(PLAN)
        for seed in [42, 43, 44]:
            streams[seed] = {r: digest(E / 'training' / f'{r}-seed{seed}' / 'batch-stream.jsonl') for r in ROUTES}
            assert len(set(streams[seed].values())) == 1, ('Unpaired training', seed)
        for route in ROUTES:
            for seed in [42, 43, 44]:
                guard(q)
                name = f'{route}-seed{seed}'
                root = E / 'training' / name
                rows = [json.loads(x) for x in (root / 'metrics-rank0.jsonl').read_text().splitlines()]
                assert len(rows) == 12000 and rows[-1]['global_step'] == 12000 and rows[-1]['opt_step'] == 6000
                assert all(math.isfinite(float(v)) for row in rows for v in row['metrics'].values())
                assert len((root / 'batch-stream.jsonl').read_text().splitlines()) == 12000
                ckpts = list((root / 'output').glob('*/checkpoint_step_12000.safetensors'))
                assert len(ckpts) == 1
                cp[name] = digest(ckpts[0])
                for n in ['config.yaml', 'normalization_stats.npy']:
                    f = ckpts[0].parent / n
                    freeze[str(f)] = digest(f)
                pc.write(dest / 'progress.json', {'time': pc.now().isoformat(), 'full_payloads_hashed': list(cp), 'total': 9})
        for task in TASKS:
            f = E / 'scenes' / task / 'manifest.json'
            freeze[str(f)] = digest(f)
        pc.frozen(freeze)
        marker(S / 'evaluation-ready.json', {'passed': True, 'time': pc.now().isoformat(),
               'checkpoint_sha256': cp, 'training_streams': streams, 'frozen_files': freeze,
               'operational_recovery': str(PLAN)})
        code = 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        pc.write(dest / 'exit.json', {'time': pc.now().isoformat(), 'returncode': code})


def idle(row, apps):
    try:
        uuid, pci, mem, util, ecc = row
        return int(mem) < 100 and int(util) == 0 and int(ecc) == 0 and uuid not in apps
    except (ValueError, TypeError):
        return False


def advance_idle(previous, row, apps):
    return previous + 1 if idle(row, apps) else 0


def sample(slot):
    line = subprocess.check_output(['nvidia-smi', '-i', str(slot),
        '--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    row = [x.strip() for x in line.split(',')]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                   '--format=csv,noheader'], text=True, timeout=15)
    return row, apps


def queue(q, slot):
    assert socket.gethostname().split('.')[0] == 'cm009' and slot in range(1, 7)
    guard(q, True)
    dest = OUT / 'queues' / str(slot)
    marker(dest / 'started.json', identity())
    protect(remaining(q, True))
    code, reason, admitted = 1, None, False
    try:
        consecutive = 0
        while True:
            guard(q, True)
            row, apps = sample(slot)
            consecutive = advance_idle(consecutive, row, apps)
            pc.write(dest / 'progress.json', {'time': pc.now().isoformat(), 'stage': 'waiting_for_own_slot',
                     'row': row, 'idle_samples': consecutive, 'deadline': q['evaluation_deadline']})
            if consecutive >= q['idle_samples']:
                # Confirm advisory lock availability; frozen queue reacquires it and rechecks GPU state.
                try:
                    lock, row = pc.reserve(slot)
                except (BlockingIOError, AssertionError):
                    consecutive = 0
                else:
                    lock.close()
                    break
            time.sleep(q['sample_seconds'])
        guard(q, True)
        assert shutil.disk_usage(E).free > q['minimum_free_bytes']
        loadplan()
        assert json.loads((S / 'evaluation-ready.json').read_text())['passed']
        seconds = min(q['queue_timeout_seconds'], remaining(q, True))
        pc.write(dest / 'admitted.json', {'time': pc.now().isoformat(), 'gpu': row, 'timeout_seconds': seconds})
        protect(seconds)
        admitted = True
        sys.argv = [str(P / 'evaluation_queue.py'), str(slot)]
        try:
            runpy.run_path(str(P / 'evaluation_queue.py'), run_name='__main__')
        except SystemExit as exc:
            assert exc.code in (None, 0), ('Frozen queue failed', exc.code)
        assert json.loads((E / 'evaluation-slots' / str(slot) / 'exit.json').read_text())['returncode'] == 0
        code = 0
    except BaseException as exc:
        reason = repr(exc)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        pc.write(dest / 'exit.json', {'time': pc.now().isoformat(), 'returncode': code,
                 'reason': reason, 'admitted': admitted, 'no_automatic_retry': True})


def controller(q):
    check(q)
    dest = OUT / 'controller'
    marker(dest / 'started.json', identity())
    protect(remaining(q))
    code, reason = 1, None
    try:
        env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        pc.write(dest / 'progress.json', {'time': pc.now().isoformat(), 'stage': 'cpu_integrity_gate'})
        rc = pc.run_child([sys.executable, str(SCRIPT), '--gate'], env, dest / 'gate.log', q['gate_timeout_seconds'] + 90)
        assert rc == 0
        launched = []
        for slot in range(1, 7):
            guard(q, True)
            assert not (OUT / 'queues' / str(slot) / 'started.json').exists()
            command = shlex.join(['timeout', '--signal=TERM', '--kill-after=90s', str(math.ceil(remaining(q, True)) + 1),
                                 'python3', '-u', str(SCRIPT), '--queue', str(slot)])
            command += ' > ' + shlex.quote(str(OUT / f'queue-{slot}.log')) + ' 2>&1'
            remote = shlex.join(['tmux', 'new-session', '-d', '-s', f'ow-eval-firststart-20260925-{slot}', command])
            rc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                 'zifanz4@cm009.csl.illinois.edu', remote], timeout=30).returncode
            assert rc == 0, ('Ambiguous or failed launch; never repeat automatically', slot, rc)
            launched.append(slot)
            pc.write(dest / 'launched.json', {'time': pc.now().isoformat(), 'slots': launched})
        while True:
            guard(q)
            states = {}
            for slot in range(1, 7):
                f = OUT / 'queues' / str(slot) / 'exit.json'
                if f.exists(): states[str(slot)] = json.loads(f.read_text())
            pc.write(dest / 'progress.json', {'time': pc.now().isoformat(), 'stage': 'independent_slot_admission_or_evaluation',
                     'terminal_slots': states, 'evaluation_deadline': q['evaluation_deadline']})
            if len(states) == 6:
                assert all(x['returncode'] == 0 for x in states.values()), 'Full matrix incomplete; all failures retained'
                break
            assert remaining(q, True) > -120, 'Evaluation workers must be terminal by fixed deadline plus cleanup'
            time.sleep(q['sample_seconds'])
        for script in ['analyze_closed_loop.py', 'audit_closed_loop.py', 'summarize_diagnostics.py']:
            guard(q)
            pc.write(dest / 'progress.json', {'time': pc.now().isoformat(), 'stage': script})
            rc = pc.run_child([str(R / 'benchmark-env/bin/python'), str(P / script)], env,
                              dest / f'{script}.log', min(3600, remaining(q)))
            assert rc == 0, script
        pc.write(dest / 'progress.json', {'time': pc.now().isoformat(), 'stage': 'complete_audited_native295M_evaluation'})
        code = 0
    except BaseException as exc:
        reason = repr(exc)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        pc.write(dest / 'exit.json', {'time': pc.now().isoformat(), 'returncode': code, 'reason': reason})


def selftest():
    r = ['GPU-a', '0000:01:00.0', '1', '0', '0']
    assert idle(r, '')
    assert not idle(r, 'GPU-a\n')
    for field, value in [(2, '100'), (3, '1'), (4, '2'), (4, 'N/A')]:
        bad = r.copy(); bad[field] = value
        assert not idle(bad, '')
    # One busy slot does not block the other slot's independent admission counter.
    counters = {1: 0, 2: 0}
    for t in range(3):
        for slot in counters:
            counters[slot] = advance_idle(counters[slot], r, '' if slot == 1 else 'GPU-a')
    assert counters == {1: 3, 2: 0}
    print(json.dumps({'passed': True, 'idle_ecc_app_checks': True, 'independent_slot_admission': True}))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--gate', action='store_true')
    g.add_argument('--controller', action='store_true')
    g.add_argument('--queue', type=int)
    g.add_argument('--self-test', action='store_true')
    args = a.parse_args()
    if args.self_test:
        selftest()
    else:
        q = loadplan()
        if args.check: print(json.dumps(check(q)))
        elif args.gate: gate(q)
        elif args.controller: controller(q)
        else: queue(q, args.queue)
