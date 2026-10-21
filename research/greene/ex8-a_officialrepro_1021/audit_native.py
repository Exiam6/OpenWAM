"""Read-only independent native evaluation audit; never schedules GPU work."""
import argparse,datetime as dt,hashlib,json,math,pathlib,re,time
ROOT=pathlib.Path('/home/zifanz4/openwam-runtime/ex8-a_officialrepro_1021')
ANSI=re.compile(r'\x1b\[[0-9;]*m')
COUNTS=re.compile(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)')
ERRORS=['CUDA/Curobo prewarm failed:', 'planner fallback engaged:', 'error occurs !', 'Traceback (most recent call last):']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def audit():
 protocol=read(ROOT/'protocol.json');state=read(ROOT/'pilot/result.json');groups=[];errors=[]
 for folder in sorted((ROOT/'pilot').glob('demo_*')):
  log=folder/'client.log'
  if not log.exists():continue
  text=ANSI.sub('',log.read_text(errors='replace'));matches=COUNTS.findall(text);rows=[];previous=0
  for s,n,seed in matches:
   s,n,seed=map(int,[s,n,seed]);assert n==len(rows)+1;assert s-previous in [0,1]
   rows.append(dict(episode_index=n-1,scene_seed=seed,success=bool(s-previous)));previous=s
  completed={p.name for p in (folder/'policy-inputs').glob('ep*') if int(p.name[2:])<len(rows)}
  bad=[];count=0
  for name in sorted(completed):
   metas=sorted((folder/'policy-inputs'/name).glob('step_*/meta.json'))
   if not metas:bad.append(f'{name}: no debug metadata')
   for f in metas:
    m=read(f);count+=1
    for key in ['state','action']:
     x=m.get(key);ok=isinstance(x,list) and len(x)==20 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in x)
     if not ok:bad.append(f'{f}: invalid {key}')
    for cam in ['head','left','right']:
     p=f.parent/(cam+'.png')
     if not p.is_file() or p.stat().st_size<64:bad.append(str(p))
  found=[x for x in ERRORS if x in text];bad.extend(found);errors.extend(bad)
  groups.append(dict(group=folder.name,scored_episodes=len(rows),successes=previous,episodes=rows,checked_action_records=count,errors=bad,client_log_sha256=sha(log),complete=len(rows)==5))
 total=sum(g['scored_episodes'] for g in groups);success=sum(g['successes'] for g in groups)
 result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),pilot_status=state['status'],pilot_scored_episodes=total,pilot_successes=success,groups=groups,errors=errors,full_paper_reproduced=False,full_endpoint_episodes=10000,full_result_available=False,targets_pct=protocol['criterion'],source_protocol_sha256=sha(ROOT/'protocol.json'))
 if state.get('readiness_passed'):
  assert len(groups)==6 and total==30 and all(g['complete'] for g in groups) and not errors
  seconds=sum(g['seconds'] for g in state['groups']);result['measured_seconds_per_episode']=seconds/30
  result['full_gpu_hours_linear_extrapolation']=seconds/30*10000/3600
  result['startup_allowance_gpu_hours']=100*180/3600
  result['full_planning_estimate_gpu_hours']=result['full_gpu_hours_linear_extrapolation']+result['startup_allowance_gpu_hours']
  result['within_registered_200_gpu_hour_planning_ceiling']=result['full_planning_estimate_gpu_hours']<=200
  result['next_gate']='freeze_L4_manifest_and_register' if result['within_registered_200_gpu_hour_planning_ceiling'] else 'report_cost_above_registered_planning_ceiling_before_L4'
 elif state['status']=='failed':result['next_gate']='diagnose_preserved_failure_before_new_attempt'
 else:result['next_gate']='finish_fixed_30_episode_readiness_pilot'
 out=ROOT/'independent-audit';out.mkdir(exist_ok=True);write(out/'status.json',result)
 # This is a completeness manifest, not authorization to launch.
 rows=[dict(task=t,mode=m,episodes=100,seed=0,status='not_registered',source='official_native_single_eval') for m in protocol['full_modes'] for t in protocol['full_tasks']]
 expected=dict(kind='planned_full_endpoint_not_dispatch_authorization',rows=rows,planned_episodes=sum(r['episodes'] for r in rows),pilot_episodes_excluded=True)
 write(out/'expected-full-endpoint.json',expected)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  r=audit()
  print(json.dumps({k:r[k] for k in ['updated','pilot_status','pilot_scored_episodes','pilot_successes','next_gate']}),flush=True)
  if not a.watch or r['pilot_status'] in ['pilot_complete','failed']:break
  if dt.datetime.now(dt.timezone.utc)>dt.datetime.fromisoformat(read(ROOT/'protocol.json')['deadline']):break
  time.sleep(30)
