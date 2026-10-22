"""CPU admission; native child starts only inside existing dispatcher."""
import argparse,datetime as dt,hashlib,json,os,pathlib,subprocess,sys
from native_adapter import BLOCKED
R=pathlib.Path('/home/zifanz4/openwam-runtime');F=R/'ex8-l_headlessprefix_1022';N=R/'nightly-gpu'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def admit():
 p=read(F/'protocol.json')
 assert p['kind']=='FROZEN_NATIVE_HEADLESS_PREFIX' and p['registered_for_dispatch']
 assert p['max_seconds']==150 and p['probe_alarm_seconds']==70
 assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
 assert not any(pathlib.Path(x).exists() for x in p['pause_paths'])
 assert not (F/'result.json').exists()
 for name,digest in p['input_sha256'].items():assert sha(name)==digest,name
 inv=read(N/'latest.json');age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(inv['time'])).total_seconds()
 assert 0<=age<25 and not inv['errors'],'Inventory stale or incomplete'
 peers=[c['id'] for x in (N/'claims').glob('*.json') for c in [read(x)] if c['id'].startswith('ex8') and c['id']!='ex8l-native-headless-prefix' and c['status'] not in ['complete','failed']]
 assert not peers,peers
 assert not any(k.startswith('ROBOTWIN_LABTASKER_') for k in os.environ)
 assert not BLOCKED.intersection(os.environ)
 return p
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--check-admission',action='store_true');v=a.parse_args();p=admit()
 if v.check_admission:print('Headless diagnostic admitted; idle30min remains dispatcher-only')
 else:
  env=dict(os.environ,**p['required_environment'])
  from prewarm import native_environment
  assert all(env[k]==v for k,v in native_environment(R,F).items())
  assert not any(pathlib.Path(x).exists() for x in p['pause_paths'])
  subprocess.run([str(R/'benchmark-env/bin/python'),str(F/'code/probe_candidate.py')],env=env,check=True,timeout=90)
