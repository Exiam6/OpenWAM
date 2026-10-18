"""CPU-only, one-shot dispatch of the already registered per-seed VAE pilots.

The frozen run_batch parent owns GPU locks, fresh capacity checks, time limits,
pause handling, process cleanup and scientific claims. This observer never
retries an attempt and never releases the subsequent full evaluation.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
CORE = ROOT / 'corrected-protocol-v1'
ENGINE = ROOT / 'corrected-engine-v1'
SEEDS = (42, 43, 44)
GPU_ORDER = (3, 1, 2, 4)
REQUIRED_SECONDS = 11760


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def prior_attempts(root, seed):
    root = Path(root)
    paths = [root / 'evaluation' / f'seed{seed}' / 'pilot',
             root / 'phase-claims' / f'pilot_evaluation-seed{seed}.json']
    paths.extend(p for p in (root / 'batches').glob('pilot_evaluation-*')
                 if str(seed) in p.name.split('-')[1:])
    return [str(p) for p in paths if p.exists()]


def training_evidence(root, seed, plan):
    """A final child result alone is insufficient: its bounded parent must pass."""
    root = Path(root)
    claim_path = root / 'phase-claims' / f'training-seed{seed}.json'
    if not claim_path.exists():
        return None
    claim = read(claim_path)
    if claim.get('status') in ('reserved', 'running'):
        return None
    assert claim.get('complete') is True and claim.get('exit_code') == 0, 'Training parent failed'
    assert claim['seed'] == seed
    run = root / 'runs' / f'vae48_native-seed{seed}'
    result_path = run / 'result.json'
    assert Path(claim['result_path']).resolve() == result_path.resolve()
    assert sha(result_path) == claim['result_sha256']
    result = read(result_path)
    assert result['seed'] == seed and result['stage'] == 'full'
    assert result['complete'] and result['steps'] == 20000 and result['strict_reload']
    assert Path(result['checkpoint']).resolve() == (run / 'checkpoint_step_20000.safetensors').resolve()
    assert Path(result['checkpoint']).is_file()
    assert result['corrected_protocol_sha256'] == plan['protocol_sha256']
    assert result['runtime_manifest_sha256'] == plan['runtime_manifest_sha256']
    init_path = run / 'initialization.json'
    assert sha(init_path) == plan['initialization_sha256'][str(seed)]
    # The unchanged GPU parent independently rehashes the full checkpoint and
    # all frozen source/assets before any GPU claim or evaluation is started.
    return dict(training_result=str(result_path), result_sha256=sha(result_path),
                parent_claim=str(claim_path), parent_claim_sha256=sha(claim_path),
                initialization_sha256=sha(init_path), checkpoint_sha256=result['checkpoint_sha256'])


def window_open(now, deadline):
    end = dt.datetime.fromisoformat(deadline) - dt.timedelta(seconds=60)
    return (end - now).total_seconds() > REQUIRED_SECONDS


def lock_available(root, gpu):
    with (Path(root) / f'gpu-{gpu}.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def choose_gpu(samples, reserved, available):
    for gpu in GPU_ORDER:
        if gpu not in reserved and samples[gpu]['idle'] and available(gpu):
            return gpu
    return None


def command(seed, gpu):
    assert seed in SEEDS and gpu in GPU_ORDER
    return [sys.executable, '-u', str(CORE / 'run_batch.py'), '--phase',
            'pilot_evaluation', '--seeds', str(seed), '--gpus', str(gpu)]


def verify_pilot_completion(root, seed, returncode):
    root = Path(root)
    assert returncode == 0, 'Pilot batch failed; preserve without retry'
    batch = read(root / 'batches' / f'pilot_evaluation-{seed}' / 'result.json')
    assert batch['status'] == 'complete' and batch['complete'] and batch['seeds'] == [seed]
    claim_path = root / 'phase-claims' / f'pilot_evaluation-seed{seed}.json'
    claim = read(claim_path)
    assert claim['complete'] and claim['exit_code'] == 0 and claim['seed'] == seed
    result_path = root / 'evaluation' / f'seed{seed}' / 'pilot' / 'result.json'
    assert Path(claim['result_path']).resolve() == result_path.resolve()
    assert sha(result_path) == claim['result_sha256']
    result = read(result_path)
    assert result['complete'] and result['status'] == 'complete'
    assert result['seed'] == seed and result['stage'] == 'pilot' and result['episodes'] == 15
    assert len(result['groups']) == 3 and all(x['episodes'] == 5 for x in result['groups'])
    return dict(result=str(result_path), result_sha256=sha(result_path),
                parent_claim_sha256=sha(claim_path), occupied_seconds=claim['occupied_seconds'])


def load_plan(expected_sha256=None):
    path = ROOT / 'pilot-capacity-queue-v1-plan.json'
    if expected_sha256 is not None:
        assert sha(path) == expected_sha256, 'Resource plan changed after admission'
    plan = read(path)
    assert plan['status'] == 'registered_resource_only' and plan['seeds'] == list(SEEDS)
    assert plan['phase'] == 'pilot_evaluation' and plan['required_remaining_seconds'] == REQUIRED_SECONDS
    assert plan['full_evaluation_automatically_released'] is False
    for path, digest in plan['files'].items():
        assert sha(path) == digest, path
    assert sha(CORE / 'protocol.json') == plan['protocol_sha256']
    assert sha(ROOT / 'corrected-engine-v1-manifest.json') == plan['runtime_manifest_sha256']
    assert str(Path(__file__).resolve()) in plan['files']
    return plan


def main(check=False):
    plan = load_plan()
    sys.path.insert(0, str(CORE))
    from corrected_protocol import load_registration, idle_capacity
    registration = load_registration()
    assert registration['evaluation_pilot_seconds'] == 3600
    assert registration['evaluation_seconds_per_seed'] == 10800
    if check:
        for seed in SEEDS:
            assert not prior_attempts(ROOT, seed)
        print(json.dumps(dict(ready=True, pending_training=[s for s in SEEDS
              if training_evidence(ROOT, s, plan) is None])))
        return
    lock = (ROOT / 'pilot-capacity-queue-v1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = ROOT / 'pilot-capacity-queue-v1'
    out.mkdir(exist_ok=False)
    state = dict(pid=os.getpid(), status='waiting_training', started=stamp(),
                 plan_sha256=sha(ROOT / 'pilot-capacity-queue-v1-plan.json'),
                 seeds={str(s): dict(status='pending') for s in SEEDS},
                 full_evaluation_automatically_released=False)
    active = {}
    stop_dispatch = False

    def persist():
        state['time'] = stamp()
        save(out / 'result.json', state)

    def stop(*_):
        nonlocal stop_dispatch
        stop_dispatch = True
        state['stop_dispatch_requested'] = stamp()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    persist()
    try:
        while True:
            for seed, child in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                row = state['seeds'][str(seed)]
                row.update(returncode=code, finished=stamp())
                try:
                    row.update(verify_pilot_completion(ROOT, seed, code), status='complete')
                except Exception as error:
                    row.update(status='failed', error=repr(error))
                    stop_dispatch = True
                del active[seed]
                persist()
            now = dt.datetime.now(dt.timezone.utc)
            paused = [p for p in registration['pause_paths'] if Path(p).exists()]
            if paused:
                stop_dispatch = True
                state['pause_paths_present'] = paused
            if not window_open(now, registration['deadline']):
                stop_dispatch = True
                state['admission_window_expired'] = stamp()
            if not stop_dispatch:
                for seed in SEEDS:
                    row = state['seeds'][str(seed)]
                    if row['status'] != 'pending':
                        continue
                    assert not prior_attempts(ROOT, seed), 'An existing pilot attempt must not be duplicated'
                    evidence = training_evidence(ROOT, seed, plan)
                    if evidence is None:
                        continue
                    samples = idle_capacity(registration, list(GPU_ORDER))
                    reserved = {state['seeds'][str(s)]['gpu'] for s in active}
                    gpu = choose_gpu(samples, reserved, lambda g: lock_available(ROOT, g))
                    if gpu is None:
                        state['status'] = 'waiting_capacity'
                        continue
                    # Revalidate immediately before launching the unchanged,
                    # independently bounded parent. It repeats all GPU gates.
                    load_plan(state['plan_sha256'])
                    assert window_open(dt.datetime.now(dt.timezone.utc), registration['deadline'])
                    assert not any(Path(p).exists() for p in registration['pause_paths'])
                    assert not prior_attempts(ROOT, seed)
                    assert training_evidence(ROOT, seed, plan) == evidence
                    argv = command(seed, gpu)
                    row.update(status='launching', gpu=gpu, command=argv,
                               evidence=evidence, launch_requested=stamp())
                    persist()
                    with (out / f'seed{seed}.log').open('x') as log:
                        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, cwd=ENGINE, start_new_session=True,
                            env=dict(os.environ, PYTHONPATH=str(ENGINE),
                                     PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                                     OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'))
                    active[seed] = child
                    row.update(status='running', pid=child.pid)
                    state['status'] = 'pilots_running'
                    persist()
            pending = [s for s in SEEDS if state['seeds'][str(s)]['status'] == 'pending']
            if not active and (stop_dispatch or not pending):
                state['status'] = ('complete' if all(x['status'] == 'complete'
                                  for x in state['seeds'].values()) else 'stopped_incomplete')
                break
            persist()
            time.sleep(15)
    except BaseException as error:
        state.update(status='observer_failed', error=repr(error))
        raise
    finally:
        # Existing children are self-contained run_batch supervisors, each with
        # immutable claims, pause checks, allocation deadline and cleanup. A CPU
        # observer error must not terminate a healthy scientific attempt.
        state['active_bounded_parents'] = {str(s): c.pid for s, c in active.items()
                                           if c.poll() is None}
        state['finished'] = stamp()
        persist()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-readiness', action='store_true')
    main(parser.parse_args().check_readiness)
