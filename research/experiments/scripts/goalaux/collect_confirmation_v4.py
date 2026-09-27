"""Resume untouched completed attempts; narrow native null-grasp annotation."""
import argparse,datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
import collect_confirmation_v2 as ref
from null_grasp_classification_v2 import annotate,is_native_no_grasp
ROOT=ref.ROOT;O=ref.O;RUN=O/'fresh-v4';ref.RUN=RUN;TASKS=ref.TASKS;write=ref.write;sha=ref.sha

def parent():
 prior=O/'fresh-v3';plan=json.loads((O/'matched-protocol.json').read_text());launch=json.loads((RUN/'launch.json').read_text());deadline=datetime.datetime.fromisoformat(launch['collection_deadline']);oldlaunch=json.loads((O/'fresh/launch.json').read_text());assert deadline.isoformat()==oldlaunch['collection_deadline']
 assert not (RUN/'started.json').exists() and not (RUN/'manifest.json').exists();assert (prior/'exit-code.txt').read_text().strip()=='1'
 for path,h in json.loads((RUN/'implementation-freeze.json').read_text())['sha256'].items():assert sha(ROOT/path)==h,path
 old=json.loads((prior/'progress.json').read_text());assert len(old['attempts'])==59 and sum(old['accepted_counts'].values())==49;assert old['attempts'][-1]['seed']==802013 and is_native_no_grasp(old['attempts'][-1])
 attempts=[annotate(r) for r in old['attempts']];assert all(r['exit_code']==0 and r['outcome']!='infrastructure_failure' for r in attempts);accepted={t:[r for r in attempts if r['task']==t and r.get('accepted')] for t in TASKS}
 for t in TASKS:
  rr=[r for r in attempts if r['task']==t];assert [r['seed'] for r in rr]==list(range(plan['new_confirmation_seeds'][t],plan['new_confirmation_seeds'][t]+len(rr)))
  for r in accepted[t]:assert sha(Path(r['hdf5']))==r['hdf5_sha256']
 write(RUN/'started.json',{'started_at':datetime.datetime.now().astimezone().isoformat(),'collection_deadline':deadline.isoformat(),'retained_attempts':59,'retained_accepted':49,'no_completed_seed_repeated':True});origin=datetime.datetime.fromisoformat(oldlaunch['started_at'])
 write(RUN/'progress.json',{'attempts':attempts,'accepted_counts':{t:len(v) for t,v in accepted.items()},'elapsed_seconds':(datetime.datetime.now().astimezone()-origin).total_seconds(),'collection_deadline':deadline.isoformat()})
 for task in TASKS:
  if len(accepted[task])==20:continue
  first=len([r for r in attempts if r['task']==task])
  for offset in range(first,40):
   for p in [O/'paused.json',ROOT/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),p
   remaining=(deadline-datetime.datetime.now().astimezone()).total_seconds();assert remaining>0,'Original collection deadline reached';seed=plan['new_confirmation_seeds'][task]+offset;dest=RUN/task/f'seed-{seed}';assert not dest.exists() and not (prior/task/f'seed-{seed}').exists();dest.mkdir(parents=True)
   with (dest/'worker.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,'-u',__file__,'--task',task,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(dest/'owned-worker.json',{'pid':proc.pid,'process_group':proc.pid,'task':task,'seed':seed})
    try:code=proc.wait(timeout=min(600,remaining))
    except subprocess.TimeoutExpired:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
     code=124
   path=dest/'result.json';raw=json.loads(path.read_text()) if path.exists() else {'task':task,'seed':seed,'accepted':False,'outcome':'infrastructure_failure','error':'worker produced no result'};raw['exit_code']=code;res=annotate(raw);attempts.append(res)
   if res.get('classification_amendment'):write(dest/'classification-amendment.json',res)
   if res.get('accepted'):accepted[task].append(res)
   write(RUN/'progress.json',{'attempts':attempts,'accepted_counts':{t:len(v) for t,v in accepted.items()},'elapsed_seconds':(datetime.datetime.now().astimezone()-origin).total_seconds(),'collection_deadline':deadline.isoformat()});print(task,seed,res['outcome'],{t:len(v) for t,v in accepted.items()},flush=True)
   if code or res['outcome']=='infrastructure_failure':raise RuntimeError('Unclassified failure retained; no retry')
   if len(accepted[task])==20:break
 write(RUN/'manifest.json',{'protocol':'first20expert-feasible scenes within40consecutive seeds per task','failed_setup_lineage':'../fresh/adjust_bottle/seed-800000/result.json','classifier_amendment':'../NULL_GRASP_ALL_CALLS_AMENDMENT.md','retained_prior_attempts':59,'attempts':attempts,'accepted':accepted,'complete':all(len(v)==20 for v in accepted.values()),'no_learned_policy_scoring':True,'elapsed_seconds':(datetime.datetime.now().astimezone()-origin).total_seconds(),'collection_deadline':deadline.isoformat()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS);p.add_argument('--seed',type=int);a=p.parse_args()
 if a.task:ref.worker(a.task,a.seed)
 else:parent()
