"""Read existing GPU02 artifacts over SSH; audit on CPU without launching jobs.

Usage: python3 research/greene/scripts/audit_expanded_results.py --output FILE
Checkpoint hashes are recorded from training reports, not recomputed here.
"""

import argparse
import collections
import itertools
import json
import math
import subprocess
from pathlib import Path

REMOTE = r'''
import datetime, hashlib, json, pathlib
p = pathlib.Path('/home/zifanz4/openwam-runtime/rae-policy-expanded-20261006')
names = ['protocol.json', 'eval-protocol.json', 'code-freeze.json',
         'evaluation-ledger.json', 'evaluation-result.json', 'queue-result.json']
names += [f'{stage}-gpu{gpu}-{suffix}.json' for gpu in (1, 2)
          for stage, suffix in [('evaluate', 'exit'), ('orchestrate', 'state')]]
files = {n: json.loads((p/n).read_text()) for n in names}
hashes = {n: hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names}
freeze = {n: {'expected': h, 'actual': hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()}
          for n, h in files['code-freeze.json']['files'].items()}
raw = {}
for r in files['evaluation-ledger.json']['records']:
    path = pathlib.Path(r['path'])
    if not path.is_relative_to(p/'evaluation'):
        raise ValueError('Result path outside study')
    data = path.read_bytes()
    raw[str(path)] = {'sha256': hashlib.sha256(data).hexdigest(), 'record': json.loads(data)}
training = [json.loads(f.read_text()) for f in sorted((p/'runs').glob('*/result.json'))]
print(json.dumps({'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'files': files, 'hashes': hashes, 'freeze': freeze, 'raw': raw, 'training': training}))
'''


def exact_sign_p(wins, losses):
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(snapshot):
    files = snapshot['files']
    protocol, evaluation = files['protocol.json'], files['eval-protocol.json']
    rows = files['evaluation-ledger.json']['records']
    def identity(r):
        return (r['scene_set'], r['route'], r['training_seed'], r['task'], r['scene_seed'], r['condition'])

    keys = [identity(r) for r in rows]
    expected = {(s, arm, seed, task, scene, cond)
                for s, tasks in evaluation['scene_sets'].items()
                for task, scenes in tasks.items() for scene in scenes
                for arm, seed, cond in itertools.product(protocol['variants'], protocol['seeds'], evaluation['conditions'])}
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Missing, extra or duplicate identities')
    require(files['evaluation-result.json']['finished'] is True and
            files['evaluation-result.json']['cells_terminal'] == files['evaluation-result.json']['cells_planned'] == 18,
            'Evaluation queue incomplete')
    for field, value in [('attempted', len(rows)),
                         ('valid', sum(r['status'] == 'complete' for r in rows)),
                         ('technical_failures', sum(r['status'] == 'infrastructure_failure' for r in rows))]:
        require(files['evaluation-ledger.json'][field] == value, f'Ledger total mismatch: {field}')
    require(all(x['actual'] == x['expected'] for x in snapshot['freeze'].values()), 'Frozen source changed')
    for r in rows:
        require({k: v for k, v in r.items() if k not in ('path', 'scene_set')} == snapshot['raw'][r['path']]['record'], 'Raw/ledger mismatch')
        require(r['status'] in ('complete', 'infrastructure_failure'), 'Unfinished result')
        if r['status'] == 'complete':
            require(type(r['success']) is bool, 'Non-boolean success')
            require(0 < r['steps'] <= r['step_limit'], 'Invalid step count')
            require(max(r['initial_render_mae'].values()) <= 1, 'Render gate failed')
    paired = collections.defaultdict(list)
    for r in rows:
        if r['status'] == 'complete':
            paired[(r['scene_set'], r['training_seed'], r['task'], r['scene_seed'])].append(r)
    for group in paired.values():
        require(len(group) == len(protocol['variants']), 'Incomplete valid pairing')
        for field in ('initial_state', 'action_seed', 'evaluation_instruction', 'step_limit'):
            require(len({r[field] for r in group}) == 1, f'Pair mismatch: {field}')
    train = snapshot['training']
    require({(r['variant'], r['seed']) for r in train} == set(itertools.product(protocol['variants'], protocol['seeds'])) and len(train) == 9, 'Training matrix incomplete')
    require(all(r['complete'] and r['strict_reload'] and r['steps'] == protocol['steps'] for r in train), 'Training incomplete')
    for gpu in (1, 2):
        require(files[f'evaluate-gpu{gpu}-exit.json']['exit_code'] == 0, 'Evaluation exited nonzero')
        require(files[f'orchestrate-gpu{gpu}-state.json']['phase'] == 'done', 'Driver not done')
    report = {'captured_at': snapshot['captured_at'], 'source_host': 'gpu02',
              'checks': {'exact_identity_matrix': True, 'raw_matches_ledger': len(rows),
                         'frozen_hashes_match': len(snapshot['freeze']), 'paired_initial_state_rng_prompt_limits': True,
                         'training_reports_complete_and_strict_reload': len(train),
                         'checkpoint_bytes_rehashed': False},
              'artifact_hashes': snapshot['hashes'], 'frozen_hashes': snapshot['freeze'],
              'by_set': {}, 'comparisons': {}, 'training_reports': train}
    for scene_set in [*evaluation['scene_sets'], 'combined']:
        subset = [r for r in rows if scene_set == 'combined' or r['scene_set'] == scene_set]
        valid = [r for r in subset if r['status'] == 'complete']
        stats = {'attempted': len(subset), 'valid': len(valid), 'technical_failures': len(subset)-len(valid), 'success': {}}
        for arm in protocol['variants']:
            stats['success'][arm] = {}
            for task in [*evaluation['scene_sets']['screen'], 'all']:
                rs = [r for r in valid if r['route'] == arm and (task == 'all' or r['task'] == task)]
                stats['success'][arm][task] = {'successes': sum(r['success'] for r in rs), 'valid': len(rs),
                    'by_seed': {str(seed): [sum(r['success'] for r in rs if r['training_seed'] == seed), sum(r['training_seed'] == seed for r in rs)] for seed in protocol['seeds']}}
        if scene_set != 'combined':
            saved = files['evaluation-result.json']['by_set'][scene_set]
            for field in ('attempted', 'valid', 'technical_failures'):
                require(stats[field] == saved[field], f'Summary mismatch {scene_set}/{field}')
            for arm, tasks in stats['success'].items():
                for task, values in tasks.items():
                    for field, value in saved['success'][arm][task].items():
                        require(values[field] == value, f'Success summary mismatch {scene_set}/{arm}/{task}')
        report['by_set'][scene_set] = stats
        comparisons = []
        for arm, base in [('raw768_wide', 'raw768_native'), ('raw768_native', 'svae48_native'), ('raw768_wide', 'svae48_native')]:
            lookup = {(r['scene_set'], r['training_seed'], r['task'], r['scene_seed']): r['success'] for r in valid if r['route'] == base}
            pairs = [(r['success'], lookup[(r['scene_set'], r['training_seed'], r['task'], r['scene_seed'])]) for r in valid if r['route'] == arm]
            wins, losses = sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs)
            comparisons.append({'arm': arm, 'base': base, 'valid_pairs': len(pairs), 'arm_only': wins, 'base_only': losses, 'two_sided_exact_sign_p': exact_sign_p(wins, losses)})
        report['comparisons'][scene_set] = comparisons
    report['technical_failures'] = [r for r in rows if r['status'] != 'complete']
    report['records'] = [{**r, 'raw_sha256': snapshot['raw'][r['path']]['sha256']} for r in rows]
    report['limits'] = ['No checkpoint-byte rehash or model inference in this audit.',
                        'Pooled sign tests follow the study rule; repeated scenes/seeds are not independent replicates.',
                        'Screen and extended are subsets of the same previously used cohort, not fresh confirmatory tests.',
                        'Data and training steps changed together; their individual effects are not identified.']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'gpu02', 'python3', '-'], input=REMOTE, text=True, capture_output=True, check=True, timeout=120)
    report = audit(json.loads(result.stdout))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('checks', 'by_set', 'comparisons')}, indent=2))


if __name__ == '__main__':
    main()
