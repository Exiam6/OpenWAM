"""CPU admission and launcher; GPU work is confined to the dispatcher child."""
import argparse,datetime as dt,hashlib,json,os,pathlib,subprocess,sys
R=pathlib.Path('/home/zifanz4/openwam-runtime');F=R/'ex8-f_renderbinding_1021';N=R/'nightly-gpu'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def peers(claims):
 return [c['id'] for c in claims if c['id'].startswith(('ex8d-','ex8e-')) and c['status'] not in ['complete','failed']]
def admit():
 p=read(F/'protocol.json');assert p['kind']=='FROZEN_NATIVE_RENDER_PROBE' and p['registered_for_dispatch']
 assert p['max_seconds']==180 and p['probe_alarm_seconds']==90
 assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
 assert not any(pathlib.Path(x).exists() for x in p['pause_paths'])
 assert not (F/'result.json').exists()
 for name,digest in p['input_sha256'].items():assert sha(name)==digest,name
 inv=read(N/'latest.json');age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(inv['time'])).total_seconds()
 assert 0<=age<25 and not inv['errors'],'Inventory stale or incomplete'
 claims=[read(x) for x in (N/'claims').glob('*.json')]
 assert not peers(claims),'Active or uncertain native peer: '+str(peers(claims))
 assert not any(k.startswith('ROBOTWIN_LABTASKER_') for k in os.environ)
 from adapter import BLOCKED
 assert not BLOCKED.intersection(os.environ),'Contaminated evaluator environment'
 return p
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--check-admission',action='store_true');v=a.parse_args();p=admit()
 if v.check_admission:print('Renderer probe admitted; GPU idle30min remains dispatcher-only')
 else:
  env=dict(os.environ,**p['required_environment'])
  subprocess.run([str(R/'benchmark-env/bin/python'),str(F/'code/probe_candidate.py'),'--registration',str(F/'protocol.json'),'--output',str(F/'result.json')],env=env,check=True,timeout=115)
