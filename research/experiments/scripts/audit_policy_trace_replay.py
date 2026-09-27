"""CPU-only replay of existing records to validate the optional trace contribution.

No model, simulator, network client, new seeds, or policy calls are constructed.
Every original trace and request file is read, hashed and left untouched.
"""
import gzip
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SOURCE = Path('/data02/zifanz4/openwam-experiments/results')
CODE = Path('/home/zifanz4/openwam-trace-contribution')
OUT = SOURCE / 'policy-trace-replay-20260921'
sys.path.insert(0, str(CODE))
from benchmarks.utils.policy_trace import TracedPolicyClient, compare_traces


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SavedResponseClient:
    def __init__(self, row):
        self.row = row
        self.response = {'action': row['server_action'], 'step': row['server_step']}
        self.calls = 0

    def predict(self, payload):
        assert hashlib.sha256(canonical(payload)).hexdigest() == self.row['request_sha256']
        self.calls += 1
        return self.response

    def close(self):
        pass


def main():
    OUT.mkdir(exist_ok=False)
    sources, records, comparisons = {}, 0, []
    for study, conditions in [('closedloop', ['clean', 'noise']), ('attribution', ['clean'])]:
        for condition in conditions:
            for seed in [400000, 400001, 400002]:
                paths = []
                for repeat in [0, 1]:
                    directory = SOURCE / ('repeatability-20260921/closedloop/' + f'r{repeat}-{condition}'
                                          if study == 'closedloop' else f'attribution-20260921/round{repeat}')
                    trace = directory / f'seed-{seed}-trace.jsonl'
                    sources[str(trace.relative_to(SOURCE))] = sha(trace)
                    rows = [json.loads(s) for s in trace.read_text().splitlines()]
                    path = OUT / f'{study}-{condition}-{seed}-r{repeat}.jsonl'
                    paths.append(path)
                    client = SavedResponseClient(rows[0])
                    with TracedPolicyClient(client, path, metadata={'study': study, 'condition': condition, 'seed': seed}) as logged:
                        for i, row in enumerate(rows):
                            assert row['step'] == i and row['seed'] == seed
                            request = directory / row['request_file']
                            sources[str(request.relative_to(SOURCE))] = sha(request)
                            payload = json.loads(gzip.decompress(request.read_bytes()))
                            client.row = row
                            client.response = {'action': row['server_action'], 'step': row['server_step']}
                            assert logged.predict(payload) is client.response
                            records += 1
                    assert client.calls == len(rows)
                    new = [json.loads(s) for s in path.read_text().splitlines()][1:]
                    for old, now in zip(rows, new):
                        assert old['request_sha256'] == now['request_sha256']
                        assert canonical(old['server_action']) == canonical(now['action'])
                        assert old['server_step'] == now['server_step']
                comparison = compare_traces(*paths)
                if study == 'closedloop':
                    for key in ['request_sha256', 'image_payload_sha256', 'state_sha256']:
                        assert comparison['first_different_event'][key] == 1
                    assert comparison['first_different_event']['action_sha256'] == 32
                    assert not comparison['same_recorded_trace']
                else:
                    assert comparison['same_recorded_trace']
                comparisons.append({'study': study, 'condition': condition, 'seed': seed, **comparison})
    assert records == 1881 and len(comparisons) == 9
    # Confirm source files remained unchanged for the duration of the replay.
    assert all(sha(SOURCE / name) == digest for name, digest in sources.items())
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'kind': 'CPU replay of saved requests/responses; no new simulation or inference',
        'code_base': subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip(),
        'code_sha256': {str(p.relative_to(CODE)): sha(p) for p in [CODE/'benchmarks/utils/policy_trace.py', CODE/'tests/test_policy_trace.py']},
        'replay_script_sha256': sha(Path(__file__)),
        'traces': 18, 'records': records, 'source_files_checked': len(sources),
        'all_original_sources_unchanged': True, 'comparisons': comparisons,
        'limitations': ['Saved responses replayed through a mock transport; no live WebSocket test.',
                        'This reproduces comparator findings, not additional independent experiments.',
                        'No evidence of a fix, global determinism, RAE superiority or control improvement.'],
    }
    (OUT/'source-manifest.json').write_text(json.dumps(sources, indent=2)+'\n')
    (OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    (OUT/'exit-code.txt').write_text('0\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['comparisons','code_sha256']},indent=2))


if __name__ == '__main__':
    main()
