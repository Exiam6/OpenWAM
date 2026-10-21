"""One separate interrupted-row attempt; native runner remains unchanged."""
import argparse,datetime as dt,importlib.util,json,pathlib,shutil,sys
R=pathlib.Path('/home/zifanz4/openwam-runtime');FULL=R/'ex8-e_rowrecovery_1021';PARENT=R/'ex8-d_nativefull_1021'
sys.path.insert(0,str(PARENT/'code'))
import run_full as native
KEY='demo_clean-adjust_bottle'
def read(p):return json.loads(p.read_text())
def account(parent,worker):
 import math
 row=next(r for r in parent['rows'] if r['key']==KEY)
 assert worker['status']=='failed' and 'Another process started using this GPU' in worker['error']
 assert worker['finished'] and 0<worker['occupied_parent_seconds']<row['max_seconds']
 return sum(r['max_seconds'] for r in parent['rows'] if r['key']!=KEY)+math.ceil(worker['occupied_parent_seconds'])+9000

def admit(key):
 assert key==KEY
 native.native_environment_guard();p=read(FULL/'protocol.json');base=read(PARENT/'protocol.json');native.validate_full_suite(base)
 assert p['kind']=='FROZEN_ONE_ROW_RESOURCE_RECOVERY' and p['registered_for_dispatch']
 assert p['rows']==[dict(next(r for r in base['rows'] if r['key']==KEY),max_seconds=9000)]
 assert p['deadline']==base['deadline']
 assert not (FULL/'runs'/key).exists()
 assert not any(pathlib.Path(x).exists() for x in p['pause_paths'])
 assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
 assert shutil.disk_usage(FULL).free>80*(1<<30)
 for name,digest in p['input_sha256'].items():assert native.sha(pathlib.Path(name))==digest,name
 for name,digest in base['input_sha256'].items():assert native.sha(pathlib.Path(name))==digest,name
 for name,digest in base['readiness_input_sha256'].items():assert native.sha(pathlib.Path(name))==digest,name
 original=native.read(native.ROOT/'protocol.json')
 for name,digest in original['input_sha256'].items():assert native.sha(pathlib.Path(name))==digest,name
 worker=read(pathlib.Path(p['parent_worker_result']))
 assert account(base,worker)==p['combined_reserved_seconds']<=base['authorized_gpu_hour_ceiling']*3600
 assert base['combined_readiness_passed'] and base['cost_gate_passed']
 return p,p['rows'][0]
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--check-admission',action='store_true');v=a.parse_args()
 if v.check_admission:admit(v.key);print('admitted')
 else:
  native.FULL=FULL;native.admit=admit;native.main(v.key)
