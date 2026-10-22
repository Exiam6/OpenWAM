"""CPU read-only service integration audit; never a native/GPU entrypoint."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'native_evidence_adapter'))
sys.path.insert(0,str(ROOT/'native_observer_worker'))
from adapter import DurableObserver, Journal
from worker import load_config, guard

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    config_path,digest,output,manifest_path=sys.argv[1:]
    config=load_config(config_path,digest)
    guard(config['pause_paths'])
    manifest=json.loads(Path(manifest_path).read_text())
    assert all(sha(p)==v for p,v in manifest['files'].items())
    result=dict(scope='REAL_READ_ONLY_CPU_TRANSIENT_SERVICE',time=dt.datetime.now(dt.timezone.utc).isoformat(),
                cgroup=Path('/proc/self/cgroup').read_text(),python=sys.version,host=socket.gethostname(),
                native_executed=False,GPU_seconds=0,native_L2_L3_passed=False,
                dispatcher_deployment_validated=False,full_release_allowed=False)
    journal=Journal(Path(output).with_suffix('.journal'))
    observer=DurableObserver([sys.executable,str(ROOT/'native_observer_worker/worker.py'),
              '--config',config_path,'--sha256',digest],journal,
              host=config['host'],device_map=config['device_map'],assigned=config['assigned'],
              own_group=config['own_group'],gdm_uid=config['gdm_uid'],seconds=4)
    try:
        result['observation']=observer()
        result['passed']=True
    except Exception as exc:
        result.update(passed=False,error=repr(exc))
    result['dependency_hashes_unchanged']=all(sha(p)==v for p,v in manifest['files'].items())
    result['journals']=[str(p) for p in journal.directory.glob('*.json')]
    guard(config['pause_paths'])
    with Path(output).open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='observation'}))
    return 0 if result['passed'] and result['dependency_hashes_unchanged'] else 1
if __name__=='__main__':raise SystemExit(main())
