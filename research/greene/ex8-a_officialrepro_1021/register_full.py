"""Fail-closed L4 registration after measured native readiness. No retries."""
import datetime as dt,hashlib,json,pathlib,shutil,subprocess,sys
R=pathlib.Path('/home/zifanz4/openwam-runtime');ROOT=R/'ex8-a_officialrepro_1021';FULL=ROOT/'full';SOURCE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
from audit_native import audit

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 assert not FULL.exists(),'Separate recovery required; never overwrite a registered full attempt'
 check=audit();assert check['pilot_status']=='pilot_complete' and check['pilot_scored_episodes']==30 and not check['errors']
 assert check['within_registered_200_gpu_hour_planning_ceiling'],check['full_planning_estimate_gpu_hours']
 pilot=read(ROOT/'pilot/result.json');assert pilot['readiness_passed']
 assert sum(g['successes'] for g in pilot['groups'] if g['mode']=='demo_clean')>0
 assert (ROOT/'independent-audit/first-episode-three-cameras.png').is_file()
 subprocess.run(['/usr/bin/python3','/home/zifanz4/openwam-vae-control/research/greene/scripts/sync_official.py'],check=True)
 base=read(ROOT/'protocol.json');sync=read(R/'official-sync/latest.json');assert sync['passed'] and sync['after']==base['official_sync']['after'],'Review new official source compatibility before registration'
 # The committed phase plan and orchestration source must exist before dispatch.
 tracked=['L4_PLAN.md','run_full.py','register_full.py','audit_native.py']
 for name in tracked:
  subprocess.run(['git','-C',str(SOURCE),'ls-files','--error-unmatch',str(SOURCE/name)],check=True,stdout=subprocess.DEVNULL)
  assert not subprocess.check_output(['git','-C',str(SOURCE),'diff','HEAD','--',str(SOURCE/name)],text=True)
 limits={}
 for line in (ROOT/'RoboTwin/task_config/_eval_step_limit.yml').read_text().splitlines():
  if line.strip() and not line.lstrip().startswith('#'):
   k,v=line.split(':',1);limits[k.strip()]=int(v)
 rows=[];weight=2*sum(limits.get(t,1000) for t in base['full_tasks'])
 for mode in base['full_modes']:
  for index,task in enumerate(base['full_tasks']):
   rows.append(dict(key=f'{mode}-{task}',task=task,mode=mode,episodes=100,seed=0,native_step_limit=limits.get(task,1000),max_seconds=600+int(660000*limits.get(task,1000)/weight)))
 assert len(rows)==100 and sum(x['max_seconds'] for x in rows)<=720000
 FULL.mkdir();(FULL/'code').mkdir();(FULL/'runs').mkdir()
 inputs={}
 for name in tracked:
  target=FULL/'code'/name;shutil.copyfile(SOURCE/name,target);inputs[str(target)]=sha(target)
 now=dt.datetime.now(dt.timezone.utc)
 p=dict(name='ex8-a_officialrepro_1021/L4',registered=now.isoformat(),deadline=(now+dt.timedelta(days=7)).isoformat(),source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),pilot_protocol_sha256=sha(ROOT/'protocol.json'),completed_pilot_sha256=sha(ROOT/'pilot/result.json'),input_sha256=inputs,rows=rows,planned_episodes=10000,reserved_gpu_hours=sum(r['max_seconds'] for r in rows)/3600,pilot_cost_audit=check,official_sync=sync,criterion=base['criterion'],retry_policy='none; retain incomplete attempts',debug_recording=False,full_paper_reproduced=False)
 write(FULL/'protocol.json',p)
 jobs=[]
 for index,row in enumerate(rows):
  argv=['/usr/bin/python3',str(FULL/'code/run_full.py'),'--key',row['key']];jid=f'ex8a-full-{index:03d}-{row["task"].replace("_","-")}'
  jobs.append(dict(id=jid,status='ready',scientific_identity=f'ex8-a-native-full/{row["key"]}/seed0',max_seconds=row['max_seconds'],authorization_evidence=str(FULL/'protocol.json'),authorization_sha256=sha(FULL/'protocol.json'),pause_paths=[str(FULL/'paused.json'),str(ROOT/'paused.json'),'/home/zifanz4/.local/state/openwam-selfcheck/paused.json'],host_commands={'gpu01':dict(argv=argv,admission_argv=argv+['--check-admission'],cwd=str(ROOT),input_sha256={str(FULL/'protocol.json'):sha(FULL/'protocol.json')},outputs_absent=[str(FULL/'runs'/row['key'])])}))
 sys.path.insert(0,str(R/'nightly-gpu/code-v3'));from policy import validate_job
 for job in jobs:
  validate_job(job);assert not (R/'nightly-gpu/jobs'/f'{job["id"]}.json').exists();assert not (R/'nightly-gpu/claims'/f'{job["id"]}.json').exists()
 write(FULL/'registry.json',dict(stage='registered_L4',protocol_sha256=sha(FULL/'protocol.json'),jobs=jobs))
 for job in jobs:write(R/'nightly-gpu/jobs'/f'{job["id"]}.json',job)
 print(json.dumps(dict(registered=len(jobs),episodes=10000,reserved_gpu_hours=p['reserved_gpu_hours'])))
if __name__=='__main__':main()
