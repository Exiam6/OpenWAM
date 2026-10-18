from pathlib import Path
import sys,json,os,datetime,fcntl,time,signal
R=Path('/home/zifanz4/openwam-runtime');G=R/'ex4-g_calibratedfactorial_1017';V=R/'ex5-a_vaebaseline_1018';D=Path(__file__).parent
sys.path.insert(0,str(G/'joint-resource-code-v3'));import joint_resources as j
sys.path.insert(0,str(V/'code'));from stage_runner import run_child
plan=j.read(D/'plan.json');assert plan['scope']=='diagnostic_only_no_release' and plan['seconds_limit']==600
assert j.sha(D/'diagnostic.py')==plan['diagnostic_code_sha256']
for path,digest in plan['frozen_files'].items():assert j.sha(path)==digest,path
assert not (V/'full-training-release.json').exists()
registration=j.read(V/'gpu-registration.json')
assert j.read(V/'dedicated-preflight-run.json')['status']=='failed'
assert j.read(V/'dedicated-preflight-run.json')['occupied_seconds']+600<registration['max_occupied_seconds']
lock=(V/'gpu-1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(pid=os.getpid(),status='checking_capacity',diagnostic_only=True,plan_sha256=j.sha(D/'plan.json'),started=j.now().isoformat());target=D/'execution.json';assert not target.exists()
def save():j.write(target,state)
def stop(*args):raise RuntimeError('Diagnostic parent terminated')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
save()
try:
 samples=[]
 for index in range(4):
  sample=j.capacity([1],registration)[1];samples.append(sample);assert sample['idle'],'GPU1 not idle; no diagnostic launched'
  assert all(not Path(p).exists() for p in registration['pause_paths'])
  if index<2:time.sleep(5)
 state.update(capacity_samples=samples,status='running');save()
 def on_started(pid):state['child_pid']=pid;save()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=sample['uuid'],PYTHONPATH=str(V/'code')+':/home/zifanz4/openwam-rae-research',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
 result=run_child([str(R/'policy-env/bin/python'),'-u',str(D/'diagnostic.py')],log=D/'run.log',seconds=600,deadline=datetime.datetime.fromisoformat(plan['deadline']),pause_paths=registration['pause_paths'],env=env,cwd='/home/zifanz4/openwam-rae-research',on_started=on_started)
 for path,digest in plan['frozen_files'].items():assert j.sha(path)==digest,path
 assert not (V/'full-training-release.json').exists()
 state.update(status='complete' if result['complete'] else 'failed',execution=result,original_evidence_unchanged=True,full_training_released=False,finished=j.now().isoformat());save()
finally:lock.close()
