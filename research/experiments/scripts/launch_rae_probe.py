"""One-shot bounded resource gate for the frozen training-only interface check."""
import datetime, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path
r = Path('/data02/zifanz4/openwam-experiments')
b = r / 'results/rae-baseline-20260922'
p = json.loads((b/'interface-protocol.json').read_text())
deadline = datetime.datetime.fromisoformat(p['deadline'])
now = datetime.datetime.now().astimezone()
assert now < deadline
assert not (b/'probe-launch.json').exists(), 'No automatic rerun'
for folder in [b, r/'results/layers-20260922', Path('/home/zifanz4/openwam-experiments/studies/layers-20260922'), Path('/home/zifanz4/openwam-experiments/studies/rae-baseline-20260922'), Path('/home/zifanz4/.local/state/openwam-selfcheck')]:
    assert not (folder/'paused.json').exists(), folder
freeze = json.loads((b/'probe-freeze.json').read_text())
for name, expected in freeze['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
source = Path('/home/zifanz4/openwam-experiments/scripts/rae_interface_preflight.py')
assert source.read_bytes() == (r/'scripts/rae_interface_preflight.py').read_bytes()
assert hashlib.sha256((b/'matched-probe-protocol.json').read_bytes()).hexdigest() == freeze['protocol_sha256']
uuid = p['resource_gate']['candidate_uuid']
lock = (r/'results/gpu-glacier-7.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
query = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,name,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total', '--format=csv,noheader,nounits'], text=True).strip()
fields = [x.strip() for x in query.split(',')]
assert fields[0] == uuid and int(fields[2]) < 100 and int(fields[3]) == 0 and int(fields[4]) == 0, query
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
assert uuid not in apps, apps
seconds = min(1200, int((deadline-datetime.datetime.now().astimezone()).total_seconds())-5)
assert seconds > 30
info = {'started_at': datetime.datetime.now().astimezone().isoformat(), 'host': os.uname().nodename, 'gpu': query, 'timeout_seconds': seconds, 'protocol_sha256': freeze['protocol_sha256'], 'deadline': p['deadline']}
(b/'probe-launch.json').write_text(json.dumps(info, indent=2)+'\n')
env = dict(os.environ, LAYER_RUN=str(r/'results/layers-20260922'), PYTHONHOME=str(r/'assets/python310-runtime'), PYTHONPATH=str(r/'env/lib/python3.10/site-packages'), WAM_ROOT=str(r), CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
with (b/'probe-run.log').open('x') as log:
    result = subprocess.run(['timeout', '--signal=TERM', '--kill-after=20s', str(seconds)+'s', str(r/'env/bin/python'), '-u', str(r/'scripts/rae_matched_probe.py')], env=env, stdout=log, stderr=subprocess.STDOUT)
end = {'finished_at': datetime.datetime.now().astimezone().isoformat(), 'exit_code': result.returncode}
(b/'probe-exit.json').write_text(json.dumps(end, indent=2)+'\n')
print(json.dumps(dict(info, **end)), flush=True)
