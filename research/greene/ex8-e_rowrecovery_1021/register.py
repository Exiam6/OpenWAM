"""Freeze a single separate recovery, then submit only to existing dispatcher."""
import datetime as dt,gzip,json,pathlib,shutil,subprocess,sys,math
from run_recovery import account,KEY,R
S=pathlib.Path(__file__).resolve().parent;E=R/'ex8-e_rowrecovery_1021';D=R/'ex8-d_nativefull_1021';N=R/'nightly-gpu'
sys.path.insert(0,str(D/'code'));from run_full import sha,classify_full,score,validate_full_suite

def read(p):return json.loads(p.read_text())
def main():
 base=read(D/'protocol.json');validate_full_suite(base)
 pauses=base['pause_paths']+[str(E/'paused.json')]
 def paused():assert not any(pathlib.Path(x).exists() for x in pauses)
 def write(p,v):
  paused();t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
 paused();assert not E.exists();assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(base['deadline'])
 worker=N/'runs/ex8d-full-000-adjust-bottle/result.json';state=D/'runs'/KEY/'result.json';w=read(worker);r=read(state)
 assert r['status']=='failed' and r['error']=="RuntimeError('Resource interruption; retain partial attempt')"
 log=D/'runs'/KEY/'client.log';assert score(log.read_text())==r['episodes'] and len(r['episodes'])==23
 assert not classify_full(log.read_text(),'adjust_bottle',True)['fatal']
 assert read(N/'claims/ex8d-full-000-adjust-bottle.json')['status']=='failed'
 assert read(D/'stage-state.json')['status']=='remaining_rows_released'
 total=account(base,w);assert total<=280*3600
 sync=read(R/'official-sync/latest.json');assert sync['passed'] and sync['after']==base['official_sync']['after'] and not sync['changed_files']
 assert (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(sync['time'])).total_seconds()<3600
 files=['PLAN.md','run_recovery.py','audit_endpoint.py','test_recovery.py','register.py']
 for name in files:
  subprocess.run(['git','-C',str(S),'ls-files','--error-unmatch',str(S/name)],check=True,stdout=subprocess.DEVNULL)
  assert not subprocess.check_output(['git','-C',str(S),'diff','HEAD','--',str(S/name)],text=True)
 observations=[]
 with gzip.open(N/'observations/2026-10-21.jsonl.gz','rt') as f:
  for line in f:
   v=json.loads(line)
   if '2026-10-21T18:37:50'<=v['time']<='2026-10-21T18:39:30':observations.append(dict(time=v['time'],gpu01_index2=v['hosts']['gpu01']['2']))
 assert any(x['user']!='zifanz4' for o in observations for x in o['gpu01_index2']['processes'])
 paused();E.mkdir();(E/'code').mkdir();(E/'runs').mkdir();(E/'evidence').mkdir()
 inputs={str(D/'protocol.json'):sha(D/'protocol.json'),str(worker):sha(worker),str(state):sha(state),str(log):sha(log)}
 for name in files:
  paused();target=E/'code'/name;shutil.copyfile(S/name,target);inputs[str(target)]=sha(target)
 for name,digest in base['input_sha256'].items():assert sha(pathlib.Path(name))==digest;inputs[name]=digest
 write(E/'evidence/resource-diagnosis.json',dict(parent_worker=w,parent_result=r,native_classifier_fatal=[],observations=observations,diagnosis='Shared GPU guard correctly stopped only our job after another user graphics process appeared; native partial23 excluded'))
 inputs[str(E/'evidence/resource-diagnosis.json')]=sha(E/'evidence/resource-diagnosis.json')
 row=dict(next(x for x in base['rows'] if x['key']==KEY),max_seconds=9000)
 selector=[dict(key=x['key'],result=str((E if x['key']==KEY else D)/'runs'/x['key']/'result.json')) for x in base['rows']]
 p=dict(name=E.name,kind='FROZEN_ONE_ROW_RESOURCE_RECOVERY',registered_for_dispatch=True,source_commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip(),registered=dt.datetime.now(dt.timezone.utc).isoformat(),deadline=base['deadline'],rows=[row],planned_episodes=100,official_sync=sync,compatibility='Same official commit; frozen native source reused unchanged',parent_protocol=str(D/'protocol.json'),parent_worker_result=str(worker),original_parent_cap_seconds=10458,parent_occupied_seconds_rounded_up=math.ceil(w['occupied_parent_seconds']),max_seconds=9000,combined_reserved_seconds=total,combined_reserved_gpu_hours=total/3600,unchanged_ceiling_gpu_hours=280,input_sha256=inputs,pause_paths=pauses,endpoint_selector=selector,excluded_attempts=[dict(result=str(state),reason='Resource interruption; all23 partial episodes excluded from complete endpoint',episodes=23)],retry_policy='One separate registered attempt; no automatic retry',full_endpoint_complete=False,full_paper_reproduced=False)
 write(E/'protocol.json',p)
 jid='ex8d-full-000-adjust-bottle-recovery';argv=['/usr/bin/python3',str(E/'code/run_recovery.py'),'--key',KEY]
 job=dict(id=jid,status='ready',scientific_identity=f'{E.name}/native-full-resource-recovery/{KEY}/seed0/attempt1',max_seconds=9000,authorization_evidence=str(E/'protocol.json'),authorization_sha256=sha(E/'protocol.json'),pause_paths=pauses,host_commands={'gpu01':dict(argv=argv,admission_argv=argv+['--check-admission'],cwd=str(E),input_sha256={str(E/'protocol.json'):sha(E/'protocol.json')},outputs_absent=[str(E/'runs'/KEY)])})
 sys.path.insert(0,str(N/'code-v3'));from policy import validate_job
 validate_job(job);assert not (N/'claims'/f'{jid}.json').exists() and not (N/'jobs'/f'{jid}.json').exists()
 write(E/'registry.json',dict(jobs=[job],protocol_sha256=sha(E/'protocol.json'),endpoint_selector=selector,publication='Local only; no external upload'))
 subprocess.run(argv+['--check-admission'],check=True,timeout=60)
 subprocess.run(['/usr/bin/python3',str(E/'code/audit_endpoint.py'),'--output',str(E/'readout.json')],check=True,timeout=60,stdout=subprocess.DEVNULL)
 paused();write(N/'jobs'/f'{jid}.json',job)
 print(json.dumps(dict(submitted=jid,reserved_gpu_hours=total/3600,deadline=p['deadline'],original_failure_retained=True)))
if __name__=='__main__':main()
