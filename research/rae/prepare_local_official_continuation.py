"""Freeze one new local official reproduction continuation without dispatching it.

The October 30 plan's Hestia probe uses scene manifests that are not on this
host. This registration continues the separately validated native full-suite
protocol already present on gpu01/gpu02. It preserves the previous stage and
all outcomes, including incomplete attempts.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


HOME = Path('/home/zifanz4')
RUNTIME = HOME / 'openwam-runtime'
OLD = RUNTIME / 'ex8-d_nativefull_1021'
NEW = RUNTIME / 'ex8-d2_nativefull_1030'
BASE = RUNTIME / 'ex8-a_officialrepro_1021'
NIGHT = RUNTIME / 'nightly-gpu'
PAUSE = HOME / '.local/state/openwam-selfcheck/paused.json'
PLAN_COMMIT = '6d479d2b34a3ac15276d8f1de71ad4af0d218130'
PREVIOUS_OFFICIAL = '782e3133f674449e2986754cfc5f7c4e204835a2'
TARGET = 'demo_randomized-adjust_bottle'


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def write_new(path: Path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(HOME / 'openwam-official-main'), *args], text=True).strip()


def main() -> None:
    assert PAUSE.exists(), 'Keep automatic dispatch paused during CPU preparation'
    assert not NEW.exists(), 'One-shot stage identity must never be overwritten'
    sync = read(RUNTIME / 'official-sync/latest.json')
    assert sync['passed'] and sync['after'] == git('rev-parse', 'HEAD')
    assert dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(sync['time']) < dt.timedelta(minutes=30)
    baseline = read(BASE / 'protocol.json')
    assert baseline['official_sync']['after'] == PREVIOUS_OFFICIAL
    previous = read(OLD / 'protocol.json')
    assert previous['full_registered_for_dispatch'] and previous['planned_episodes'] == 10000
    assert previous['authorized_gpu_hour_ceiling'] == 280
    assert previous['checkpoint_sha256'] == baseline['checkpoint_sha256']
    assert sum(row['max_seconds'] for row in previous['rows']) <= 280 * 3600
    for name, expected in {**previous['input_sha256'], **previous['readiness_input_sha256']}.items():
        assert digest(Path(name)) == expected, f'Previous evidence drift: {name}'
    coverage = read(HOME / '.local/state/openwam-selfcheck/evidence/20261023T1830Z/composite-coverage.json')
    assert coverage['validated_accepted_episodes'] == 500
    assert TARGET not in {row['key'] for row in coverage['rows']}
    row = next(row for row in previous['rows'] if row['key'] == TARGET)
    assert row['episodes'] == 100 and row['seed'] == 0 and row['max_seconds'] < 9000
    changed = git('diff', '--name-only', f'{PREVIOUS_OFFICIAL}..{sync["after"]}').splitlines()
    affected = [name for name in changed if name.startswith(('benchmarks/robotwin/', 'scripts/deploy.py',
                                                               'openwam/model', 'openwam/policy', 'openwam/inference'))]
    assert not affected, f'Native benchmark compatibility requires review: {affected}'
    assert subprocess.check_output(['git', '-C', str(HOME / 'openwam-plan-20261030'),
                                    'rev-parse', 'HEAD'], text=True).strip() == PLAN_COMMIT
    assert shutil.disk_usage(RUNTIME).free > 100 * (1 << 30)
    NEW.mkdir()
    (NEW / 'code').mkdir()
    (NEW / 'runs').mkdir()
    for source in ('classify_log.py', 'run_full.py'):
        text = (OLD / 'code' / source).read_text()
        if source == 'run_full.py':
            old = "FULL=R/'ex8-d_nativefull_1021'"
            assert text.count(old) == 1
            text = text.replace(old, "FULL=R/'ex8-d2_nativefull_1030'")
        (NEW / 'code' / source).write_text(text)
    now = dt.datetime.now(dt.timezone.utc)
    deadline = now + dt.timedelta(days=7)
    plan = (f'# ex8-d2: local official native full continuation, {now.date()}\n\n'
            f'Primary goal: complete the released OpenWAM RoboTwin-Full 50-task clean/randomized endpoint. '
            f'This frozen stage retains the original 100 task/mode rows, 100 accepted episodes each, seed 0, '
            f'official checkpoint {baseline["checkpoint_sha256"]}, native scene, planner, renderer, '
            f'inference optimizations, scoring, and complete video validation. No prior attempt is overwritten.\n\n'
            f'First queued row: {TARGET}, exactly 100 accepted episodes, hard allocation {row["max_seconds"]} seconds. '
            f'It is absent from the independently accepted 500-episode composite. The first row must complete '
            f'and pass native logs, result count, video decoding, GPU process isolation and independent claim '
            f'validation before more rows are released. The full stage cap is 280 GPU hours and its deadline is '
            f'{deadline.isoformat()}. Only this one row is staged now.\n\n'
            f'Compatibility: official main synchronized at {sync["after"]}; its {len(changed)} changed files '
            f'since {PREVIOUS_OFFICIAL} do not touch native benchmark, deployment or inference paths. '
            f'The evaluated source remains the frozen official {PREVIOUS_OFFICIAL} in {BASE}; synchronization '
            f'does not upgrade that source. The October 30 plan commit is {PLAN_COMMIT}. '
            f'The Hestia 30-episode diagnostic remains a separate endpoint with sealed manifests unavailable '
            f'on this host.\n\n'
            f'Stop on a peer GPU process, stale inventory, deadline or budget exhaustion, missing videos, '
            f'native technical error, result mismatch, checkpoint drift, or storage reserve failure. '
            f'Retain all partial attempts and failures. GPU admission requires 1,800 seconds observed idle, '
            f'no foreign process, and the America/Chicago window for nonpriority cards.\n')
    (NEW / 'PLAN.md').write_text(plan)
    p = copy.deepcopy(previous)
    p.update(name=NEW.name, kind='FROZEN_NATIVE_FULL_CONTINUATION',
             created=now.isoformat(), registered=now.isoformat(), deadline=deadline.isoformat(),
             source_commit=previous['source_commit'], registration_plan_commit=PLAN_COMMIT,
             frozen_official_source_commit=PREVIOUS_OFFICIAL,
             prior_stage=str(OLD), first_queued_key=TARGET,
             official_sync=sync,
             compatibility={'previous_official': PREVIOUS_OFFICIAL, 'current_official': sync['after'],
                            'changed_files': changed, 'native_inference_paths_changed': affected,
                            'experiment_upgraded': False},
             continuation_accepted_manifest=str(HOME / '.local/state/openwam-selfcheck/evidence/20261023T1830Z/composite-coverage.json'),
             remaining_steps=['Finish and independently accept first native100 row',
                              'Release distinct missing rows under the GPU policy',
                              'Complete clean/randomized 10000-episode endpoint and gap analysis'],
             pause_paths=list(dict.fromkeys(previous['pause_paths'] + [str(NEW / 'paused.json')])))
    p['input_sha256'].update({str(NEW / 'PLAN.md'): digest(NEW / 'PLAN.md'),
                              str(NEW / 'code/run_full.py'): digest(NEW / 'code/run_full.py'),
                              str(NEW / 'code/classify_log.py'): digest(NEW / 'code/classify_log.py'),
                              str(OLD / 'protocol.json'): digest(OLD / 'protocol.json'),
                              str(Path(p['continuation_accepted_manifest'])): digest(Path(p['continuation_accepted_manifest']))})
    write_new(NEW / 'protocol.json', p)
    protocol_sha = digest(NEW / 'protocol.json')
    jid = 'ex8d2-full-000-randomized-adjust-bottle'
    argv = ['/usr/bin/python3', str(NEW / 'code/run_full.py'), '--key', TARGET]
    host_command = dict(argv=argv, admission_argv=argv + ['--check-admission'], cwd=str(NEW),
                        input_sha256={str(NEW / 'protocol.json'): protocol_sha},
                        outputs_absent=[str(NEW / 'runs' / TARGET)])
    job = dict(id=jid, status='ready',
               scientific_identity=f'{NEW.name}/native-full/{TARGET}/seed0/attempt1',
               max_seconds=row['max_seconds'], authorization_evidence=str(NEW / 'protocol.json'),
               authorization_sha256=protocol_sha, pause_paths=p['pause_paths'],
               host_commands={'gpu01': host_command, 'gpu02': copy.deepcopy(host_command)})
    write_new(NEW / f'{jid}.staged.json', job)
    write_new(NEW / 'FREEZE.json', dict(stage=str(NEW), status='staged_not_queued',
                                        first_row=TARGET, first_job=jid,
                                        protocol_sha256=protocol_sha, GPU_seconds=0,
                                        prior_completed_episodes=coverage['validated_accepted_episodes']))
    print(json.dumps({'stage': str(NEW), 'job': jid, 'max_seconds': row['max_seconds'],
                      'protocol_sha256': protocol_sha, 'queued': False}, indent=2))


if __name__ == '__main__':
    main()
