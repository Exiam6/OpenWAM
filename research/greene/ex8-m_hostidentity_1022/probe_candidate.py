"""Unique frozen dispatcher-only native headless prefix attempt."""
import datetime as dt,hashlib,json,os,pwd,socket,subprocess,sys,time
from pathlib import Path
from integration import NativeBackend,run,observer,DEPENDENCIES
from native_adapter import guard as native_guard
R=Path('/home/zifanz4/openwam-runtime');F=R/'ex8-l_headlessprefix_1022'
def check_pause(p):
 assert not any(Path(x).exists() for x in p['pause_paths']),'Paused'
def main():
 start=time.monotonic();result=dict(status='failed',scored_episodes=0,native_L2_L3_passed=False,full_release_allowed=False,full_paper_reproduced=False)
 backend=None
 try:
  p=json.loads((F/'protocol.json').read_text());check_pause(p)
  assert p['registered_for_dispatch'] and p['kind']=='FROZEN_NATIVE_HEADLESS_PREFIX'
  assert p['max_seconds']==150 and p['probe_alarm_seconds']==70
  assert not (F/'result.json').exists()
  for name,digest in p['input_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
  assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
  assigned=native_guard(os.environ,['--config','renderer-only-probe.yml'])
  assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(os.environ['OPENWAM_ALLOCATION_END'])-dt.timedelta(seconds=150)
  own_group=Path('/proc/self/cgroup').read_text()
  from inventory import unit_path
  unit_path(own_group)
  host=socket.gethostname();assert host==os.environ['OPENWAM_GPU_HOST']=='gpu01'
  mapping=p['device_map'];assert assigned in mapping.values()
  config=dict(host=host,device_map=mapping,assigned=assigned,own_group=own_group,gdm_uid=pwd.getpwnam('gdm').pw_uid,seconds=3,pause_paths=p['pause_paths'])
  check_pause(p)
  raw=(json.dumps(config,sort_keys=True)+'\n').encode();cp=F/'observer-config.json'
  with cp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  obs=observer(config,[sys.executable,str(DEPENDENCIES/'native_observer_worker/worker.py'),'--config',str(cp),'--sha256',hashlib.sha256(raw).hexdigest()],F/'journal',seconds=4)
  bus=subprocess.run(['/usr/bin/nvidia-smi','--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],check=True,capture_output=True,text=True,timeout=4)
  expected=dict(line.strip().split(', ') for line in bus.stdout.splitlines())[assigned]
  backend=NativeBackend(F/'code/libheadless.so',expected)
  result=run(backend,obs,lambda:check_pause(p),assigned,seconds=70,sample_timeout=14)
 except BaseException as exc:result.update(status='failed',error=repr(exc))
 finally:
  if backend is not None:result['native_details']=backend.details
  result.update(occupied_seconds_including_setup=time.monotonic()-start,scored_episodes=0,native_L2_L3_passed=False,full_release_allowed=False,full_paper_reproduced=False)
  # Retain failure evidence even if a pause arrives during the native call.
  with (F/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if result['status']=='supervisor_sequence_passed' else 1
if __name__=='__main__':raise SystemExit(main())
