"""Deploy the user's nightly gpu02 resource amendment without changing the study recipe.

Run locally with the repository's Python. Only idle study workers may be replaced;
all existing frozen artifacts are checked and archived before scheduler changes.
"""

import json
import shlex
import subprocess
from pathlib import Path

REMOTE = r"""
import datetime as dt
import fcntl
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

r = pathlib.Path('/home/zifanz4/openwam-runtime/rae-factorial-20261010')
bundle = json.load(sys.stdin)
p = json.loads((r / 'protocol.json').read_text())
deadline = dt.datetime.fromisoformat(p['deadline'])
assert dt.datetime.now(dt.timezone.utc) < deadline
assert not (r / 'resource-policy.json').exists(), 'amendment already installed'
locks = []
for gpu in range(8):
    lock = (r.parent / f'gpu-{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(lock)
for directory in ['claims', 'post-claims']:
    for path in (r / directory).glob('*.json'):
        assert json.loads(path.read_text())['status'] in ('complete', 'failed'), path
manifests = {}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for name in ['code-freeze.json', 'eval-freeze.json']:
    manifests[name] = json.loads((r / name).read_text())
    for path, expected in manifests[name]['files'].items():
        assert sha(pathlib.Path(path)) == expected, path
lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
gpus = dict(tuple(x.strip() for x in line.split(',')) for line in lines.splitlines())
assert set(gpus) == set(map(str, range(8)))
for gpu, uuid in p['gpus'].items():
    assert gpus[gpu] == uuid
units = [f'openwam-rae-factorial-{prefix}gpu{gpu}-20261010.service'
         for prefix in ['', 'post-'] for gpu in ['1', '2']]
subprocess.run(['systemctl', '--user', 'stop', *units], check=True)
subprocess.run(['systemctl', '--user', 'reset-failed', *units], capture_output=True)
archive = r / 'resource-amendment-20261011'
archive.mkdir()
for name in ['code-freeze.json', 'eval-freeze.json', 'worker.py', 'post_worker.py', 'monitor.py',
             'worker-services.json', 'post-worker-services.json']:
    shutil.copy2(r / name, archive / name)
policy = dict(host='gpu02', timezone='America/Chicago', start_hour=1, end_hour=9,
              baseline_gpus=['1', '2'], gpus=gpus, idle_only=True,
              user_message='每天晚上1am-9am 你可以占用任何空的卡',
              recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
              study_deadline=p['deadline'], max_attempt_cleanup_margin_seconds=120)
(r / 'resource-policy.json').write_text(json.dumps(policy, indent=2, ensure_ascii=False) + '\n')
for name, source in bundle.items():
    assert name in ['worker.py', 'post_worker.py', 'monitor.py']
    compile(source, name, 'exec')
    (r / name).write_text(source)
for name, manifest in manifests.items():
    for changed in [*bundle, 'resource-policy.json']:
        manifest['files'][str(r / changed)] = sha(r / changed)
    manifest['resource_amendment'] = str(archive)
    (r / name).write_text(json.dumps(manifest, indent=2) + '\n')
record = dict(policy=policy, scientific_protocol_sha256=sha(r / 'protocol.json'),
              previous_manifests={n: sha(archive / n) for n in manifests},
              new_manifests={n: sha(r / n) for n in manifests}, services=[])
for prefix, script in [('', 'worker.py'), ('post-', 'post_worker.py')]:
    services = []
    for gpu in gpus:
        unit = f'openwam-rae-factorial-{prefix}gpu{gpu}-20261010.service'
        # Hard cgroup deadline includes cleanup; no extension of the 36-hour authorization.
        remaining = int((deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()) - 60
        assert remaining > 0
        args = ['systemd-run', '--user', '--collect', '--unit=' + unit,
                '--property=RuntimeMaxSec=' + str(remaining), '--property=KillMode=control-group',
                '--property=TimeoutStopSec=45', '--property=Restart=no',
                '--working-directory=' + str(r),
                str(r.parent / 'policy-env/bin/python'), '-u', str(r / script), '--gpu', gpu]
        subprocess.run(args, check=True, capture_output=True, text=True)
        row = dict(unit=unit, gpu=gpu, deadline=p['deadline'])
        services.append(row)
        record['services'].append(row)
    name = 'post-worker-services.json' if prefix else 'worker-services.json'
    (r / name).write_text(json.dumps(services, indent=2) + '\n')
(r / 'resource-amendment.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(record, indent=2, ensure_ascii=False))
"""


def main():
    source = Path(__file__).resolve().parent
    bundle = {name: (source / name).read_text() for name in ["worker.py", "post_worker.py", "monitor.py"]}
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "gpu02", "python3 -c " + shlex.quote(REMOTE)],
        input=json.dumps(bundle),
        text=True,
        capture_output=True,
        check=True,
    )
    record = source.parent / "records/rae-factorial-resource-amendment-20261011.json"
    record.write_text(result.stdout)
    print(result.stdout)


if __name__ == "__main__":
    main()
