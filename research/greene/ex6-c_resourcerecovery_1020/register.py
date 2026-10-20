"""Freeze resource-only recovery; original complete outcomes are never rerun."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parent
R = Path('/home/zifanz4/openwam-runtime')
OLD = R / 'ex6-a_tasktransfer_1020'
ROOT = R / 'ex6-c_resourcerecovery_1020'
N = R / 'nightly-gpu'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def missing_scenes(selected, verified):
    """Success and failure outcomes both count as complete."""
    assert len(selected) == len(set(selected)) == 5
    assert set(verified) <= set(selected)
    return [s for s in selected if s not in verified]


def main():
    sync = read(R / 'official-sync/latest.json')
    assert sync['passed']
    assert (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(sync['time'])).total_seconds() < 3600
    spec = importlib.util.spec_from_file_location('transfer_audit', SOURCE.parent / 'ex6-b_contributionaudit_1020/watch_transfer.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    state = audit.snapshot(sha(OLD / 'protocol.json'))
    p = read(OLD / 'protocol.json')
    tasks = [t for t, v in state['tasks'].items() if v.get('resource_interruption')]
    assert set(tasks) == {'click_bell', 'press_stapler', 'lift_pot', 'stack_blocks_two'}
    missing = {}
    freeze = {str(OLD / 'protocol.json'): sha(OLD / 'protocol.json')}
    charged = 0
    for task in p['tasks']:
        claim_path = N / 'claims' / ('ex6a-' + task.replace('_', '-') + '.json')
        c = read(claim_path)
        assert c['status'] in ['complete', 'failed']
        charged += c['worker_result']['occupied_parent_seconds']
        freeze[str(claim_path)] = sha(claim_path)
    for task in tasks:
        cohort = OLD / 'tasks' / task / 'code/scenes'
        selected = [r['seed'] for r in read(cohort / task / 'manifest.json')['accepted']]
        missing[task] = {}
        for path in cohort.rglob('*'):
            if path.is_file():
                freeze[str(path)] = sha(path)
        for arm in p['arms']:
            rows = state['tasks'][task]['arms'][arm['name']]['rows']
            assert not any(r['status'] == 'audit_failure' for r in rows), rows
            verified = [r['scene_seed'] for r in rows if r['status'] == 'verified']
            missing[task][arm['name']] = missing_scenes(selected, verified)
            for path in (OLD / 'tasks' / task / arm['name'] / 'episodes').glob('seed-*/*'):
                if path.name in ['result.json', 'trace.jsonl']:
                    freeze[str(path)] = sha(path)
    assert sum(len(v) for arms in missing.values() for v in arms.values()) == 55
    assert charged + len(tasks)*p['job_max_seconds'] <= p['total_gpu_hour_cap']*3600
    ROOT.mkdir(exist_ok=False)
    for name in ['template', 'robot-configs']:
        shutil.copytree(OLD / name, ROOT / name)
    shutil.copy2(OLD / 'base-environment.json', ROOT / 'base-environment.json')
    for name in ['run_task.py', 'watch.py', 'register.py', 'test_recovery.py']:
        shutil.copy2(SOURCE / name, ROOT / name)
    p.update(name=ROOT.name, original_root=str(OLD), tasks={t:p['tasks'][t] for t in tasks},
             missing=missing, recovery_inputs_sha256=freeze,
             original_protocol_sha256=sha(OLD / 'protocol.json'),
             created=dt.datetime.now(dt.timezone.utc).isoformat(),
             resource_instruction='gpu01 GPUs 1/2 any hour, 30min observed idle; other cards 01:00–09:30 Chicago',
             recovery_rule='Exactly one separately registered recovery per resource-interrupted task. Only missing episodes, same fixed scenes/checkpoints/code; completed failures and successes retained unchanged.',
             previously_charged_seconds=charged, recovery_reserved_seconds=len(tasks)*p['job_max_seconds'],
             combined_worst_case_gpu_hours=(charged+len(tasks)*p['job_max_seconds'])/3600,
             official_sync=sync, official_compatibility='No official source changes since pilot freeze; scientific runtime unchanged.')
    for path in ROOT.rglob('*'):
        if path.is_file():
            p['code_sha256'][str(path)] = sha(path)
    write(ROOT / 'protocol.json', p)
    write(ROOT / 'original-audit.json', state)
    # Only after every artifact is frozen do jobs become visible to dispatch.
    for task in tasks:
        jid = 'ex6c-' + task.replace('_', '-')
        argv = ['/usr/bin/python3', str(ROOT / 'run_task.py'), '--task', task]
        job = dict(id=jid, status='ready', scientific_identity=f'{ROOT.name}/resource-recovery/{task}/seed42',
                   max_seconds=p['job_max_seconds'], authorization_evidence=str(ROOT / 'protocol.json'),
                   authorization_sha256=sha(ROOT / 'protocol.json'),
                   pause_paths=[str(ROOT / 'paused.json'), str(OLD / 'paused.json'), '/home/zifanz4/.local/state/openwam-selfcheck/paused.json'],
                   host_commands={'gpu01':dict(argv=argv, admission_argv=argv+['--check-admission'], cwd=str(ROOT),
                       input_sha256={str(ROOT/'protocol.json'):sha(ROOT/'protocol.json'), str(ROOT/'run_task.py'):sha(ROOT/'run_task.py')},
                       outputs_absent=[str(ROOT / 'tasks' / task)])})
        dest = N / 'jobs' / f'{jid}.json'
        assert not dest.exists()
        temp = dest.with_suffix('.pending')
        write(temp, job)
        temp.replace(dest)
    print(json.dumps({'missing_episodes':55,'tasks':tasks,'combined_worst_case_gpu_hours':p['combined_worst_case_gpu_hours'],'protocol_sha256':sha(ROOT/'protocol.json')}))


if __name__ == '__main__':
    main()
