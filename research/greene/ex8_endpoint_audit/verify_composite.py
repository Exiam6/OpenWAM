"""Read-only composite verification and paired-task uncertainty reporting."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import random
import re
from worker_completion import confirm_rows, select_completed

R = pathlib.Path('/home/zifanz4/openwam-runtime')
D = R / 'ex8-d_nativefull_1021'
E = R / 'ex8-e_rowrecovery_1021'
GLOBAL_PAUSE = pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')
TARGETS = dict(clean=93.74, randomized=93.46, mean=93.60)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpaused():
    protocol = read(E / 'protocol.json')
    assert not GLOBAL_PAUSE.exists(), 'Global pause'
    assert not any(pathlib.Path(x).exists() for x in protocol['pause_paths']), 'Stage pause'


def native_rows(text, final=False):
    """Parse each printed episode, separately from the frozen runner parser."""
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    rows = []
    outcome = None
    previous_successes = 0
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line in ('Success!', 'Fail!'):
            assert outcome is None, 'Multiple outcomes before native counter'
            outcome = line == 'Success!'
        elif line.startswith('Success rate:'):
            match = re.fullmatch(r'Success rate:\s*(\d+)/(\d+)\s*=>.*current seed:\s*(\d+)\s*', line)
            if not match and not final and index == len(lines)-1 and not text.endswith('\n'):
                break  # A live print may still be writing its final counter.
            assert match, 'Malformed native counter'
            successes, total, seed = map(int, match.groups())
            assert total == len(rows) + 1, 'Missing or duplicate episode'
            assert outcome is not None, 'Native outcome missing'
            assert successes - previous_successes == int(outcome), 'Outcome disagrees with counter'
            assert not rows or seed > rows[-1]['scene_seed'], 'Candidate seed not increasing'
            rows.append(dict(episode_index=total-1, scene_seed=seed, success=outcome))
            previous_successes = successes
            outcome = None
    if final:
        assert outcome is None, 'Uncounted final outcome'
    return rows


def statistics(rows, expected_tasks):
    expected = {(m, t) for m in ('demo_clean', 'demo_randomized') for t in expected_tasks}
    keys = [(r['mode'], r['task']) for r in rows]
    assert len(expected_tasks) == len(set(expected_tasks)) == 50
    assert len(keys) == len(set(keys)) == 100 and set(keys) == expected, 'Incomplete or duplicate endpoint'
    assert all(type(r['successes']) is int and 0 <= r['successes'] <= 100 and r['episodes'] == 100 for r in rows)
    clean = {r['task']: r['successes'] for r in rows if r['mode'] == 'demo_clean'}
    randomized = {r['task']: r['successes'] for r in rows if r['mode'] == 'demo_randomized'}
    c, r = sum(clean.values())/50, sum(randomized.values())/50
    rates = dict(clean=c, randomized=r, mean=(c+r)/2)
    samples = {k: [] for k in rates}
    rng = random.Random(20261021)
    tasks = sorted(expected_tasks)
    for _ in range(10000):
        draw = rng.choices(tasks, k=50)
        a, b = sum(clean[t] for t in draw)/50, sum(randomized[t] for t in draw)/50
        for k, v in dict(clean=a, randomized=b, mean=(a+b)/2).items():
            samples[k].append(v)
    gaps = {k: rates[k]-TARGETS[k] for k in rates}
    return dict(rates_pct=rates, targets_pct=TARGETS, gaps_pp=gaps,
                task_bootstrap_95pct={k: [sorted(v)[249], sorted(v)[9749]] for k,v in samples.items()},
                bootstrap_seed=20261021, bootstrap_replicates=10000,
                bootstrap_note='Paired task resampling; across-task variation only. Author seed variance unavailable.',
                local_tolerance_pp=2, practical_agreement_passed=all(abs(v)<=2 for v in gaps.values()),
                exact_published_numbers_matched=all(abs(v)<1e-8 for v in gaps.values()),
                full_paper_reproduced=False)


def audit():
    protocol = read(E/'protocol.json')
    for name, expected in protocol['input_sha256'].items():
        assert digest(pathlib.Path(name)) == expected, name
    parent = read(D/'protocol.json')
    original_path = R/'ex8-a_officialrepro_1021/protocol.json'
    assert digest(original_path) == parent['pilot_protocol_sha256']
    original = read(original_path)
    evaluator = R/'ex8-a_officialrepro_1021/RoboTwin/script/eval_policy.py'
    assert digest(evaluator) == original['input_sha256'][str(evaluator)]
    spec = importlib.util.spec_from_file_location('frozen_composite_audit', E/'code/audit_endpoint.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    result = frozen.audit()
    raw_checks = []
    for entry in protocol['endpoint_selector']:
        path = pathlib.Path(entry['result'])
        if not path.exists():
            continue
        state = read(path)
        log = path.parent/'client.log'
        if not log.exists():
            continue
        episodes = native_rows(log.read_text(), final=state['status']=='complete')
        saved = state['episodes']
        # The running supervisor may save just before the next log write.
        assert episodes[:len(saved)] == saved, entry['key']
        if state['status'] == 'complete':
            assert episodes == saved and len(episodes) == 100
        raw_checks.append(dict(key=entry['key'], status=state['status'],
                               observed_episodes=len(episodes), saved_episodes=len(saved),
                               included_in_endpoint=state['status']=='complete'))
    result.update(independent_native_outcome_checks=raw_checks,
                  audit_source_sha256=digest(pathlib.Path(__file__)),
                  native_eval_source_sha256=digest(R/'ex8-a_officialrepro_1021/RoboTwin/script/eval_policy.py'),
                  audit_kind='CPU_ONLY_FROZEN_COMPOSITE_CROSSCHECK',
                  software_difference='Benchmark torch2.7.0+cu128 vs documented2.4.1; Blackwell hardware',
                  full_paper_reproduced=False)
    confirmed, worker_checks = confirm_rows(result['per_task'], R)
    select_completed(result, confirmed, worker_checks)
    result['worker_completion_source_sha256'] = digest(pathlib.Path(__file__).with_name('worker_completion.py'))
    if result['full_endpoint_complete']:
        result.update(statistics(result['per_task'], original['full_tasks']))
    else:
        assert 'rates_pct' not in result, 'No partial headline'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    allowed = pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/evidence').resolve()
    assert destination.is_relative_to(allowed), 'Audit may only write self-check evidence'
    unpaused()
    result = audit()
    result['updated'] = dt.datetime.now(dt.timezone.utc).isoformat()
    unpaused()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.pending')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    unpaused()
    temporary.replace(destination)
    print(json.dumps({k: result[k] for k in ['updated', 'completed_rows', 'scored_episodes', 'full_endpoint_complete', 'independent_native_outcome_checks']}))
