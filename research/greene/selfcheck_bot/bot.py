"""Quarter-hour research supervision, with bounded Codex takeover on actionable gaps."""
from __future__ import annotations
import argparse,datetime as dt,fcntl,hashlib,json,os,pathlib,signal,subprocess,sys,time,traceback
from zoneinfo import ZoneInfo
SOURCE=pathlib.Path(__file__).resolve().parent
DEFAULT_STATE=pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck')
TERMINAL={'complete','failed'}
def utcnow():return dt.datetime.now(dt.timezone.utc)
def read(path,default=None):
 p=pathlib.Path(path)
 return json.loads(p.read_text()) if p.exists() else default
def write(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
def stamp(value):return dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def command(argv,timeout=20):
 p=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=timeout)
 return dict(returncode=p.returncode,stdout=p.stdout[-12000:],stderr=p.stderr[-2000:])
def service_state(unit):
 r=command(['systemctl','--user','show',unit,'--property=ActiveState,LoadState,SubState,MainPID'])
 return dict(x.split('=',1) for x in r['stdout'].splitlines() if '=' in x) | {'query_returncode':r['returncode']}
def collect(config,now=None):
 now=now or utcnow();runtime=pathlib.Path(config['runtime']);night=runtime/'nightly-gpu';snapshot=read(night/'latest.json',{})
 data=dict(last_takeover=read(pathlib.Path(config['state_dir'])/'agent.json',{}),expected_gpu_count=16,time=now.isoformat(),runtime=str(runtime),snapshot=snapshot,controller=service_state('openwam-nightly-gpu.service'),jobs=[],experiments={},read_errors=[])
 prefixes=tuple(config['job_prefixes']);data['inventory_age_seconds']=now.timestamp()-stamp(snapshot['time']) if snapshot.get('time') else None
 for path in sorted((night/'jobs').glob('*.json')):
  if not path.stem.startswith(prefixes):continue
  try:
   j=read(path);claim=read(night/'claims'/path.name);pauses=[p for p in j.get('pause_paths',[]) if pathlib.Path(p).exists()]
   authorization=pathlib.Path(j['authorization_evidence']);valid=authorization.is_file() and hashlib.sha256(authorization.read_bytes()).hexdigest()==j['authorization_sha256']
   outputs=[pathlib.Path(x) for cmd in j['host_commands'].values() for x in cmd['outputs_absent']]
   progress=[]
   for output in outputs:
    progress.extend(output.glob('server.log'));progress.extend(output.glob('client.log'));progress.extend(output.glob('*/client.log'))
   progress.append(night/'runs'/j['id']/'run.log')
   newest=max([f.stat().st_mtime for f in progress if f.exists()]+[0])
   experiment=authorization.relative_to(runtime).parts[0] if authorization.is_relative_to(runtime) else ''
   data['jobs'].append(dict(stage='native_full' if any('--key' in cmd['argv'] and any('run_full.py' in arg for arg in cmd['argv']) for cmd in j['host_commands'].values()) else 'pilot',experiment=experiment,last_log_progress_timestamp=newest,id=j['id'],scientific_identity=j['scientific_identity'],max_seconds=j['max_seconds'],hosts=list(j['host_commands']),claim=claim,created_timestamp=path.stat().st_mtime,paused_by=pauses,authorization_valid=valid))
  except Exception as e:data['read_errors'].append(dict(path=str(path),error=repr(e)))
 # Only small supervisory files and live native logs, never checkpoint contents.
 for folder in sorted(runtime.glob('ex8-*')):
  if not folder.is_dir():continue
  item=dict(path=str(folder),log_progress_timestamp=0,files={})
  for rel in ['protocol.json','pilot/result.json','independent-audit/continuation.json','readout.json','stage-state.json','full/readout.json','full/protocol.json']:
   p=folder/rel
   if not p.exists():continue
   try:
    value=read(p)
    if rel.endswith('protocol.json'):value={k:value.get(k) for k in ['name','deadline','max_seconds','planned_episodes','full_planned_episodes','criterion','full_registered_for_dispatch']}
    if isinstance(value,dict):value={k:v for k,v in value.items() if k not in ['input_sha256','jobs','per_task','pending','videos']}
    item['files'][rel]=value
   except Exception as e:data['read_errors'].append(dict(path=str(p),error=repr(e)))
  for p in list((folder/'pilot').glob('*/client.log'))+list((folder/'pilot').glob('server.log'))+list((folder/'full/runs').glob('*/client.log'))+list((folder/'full/runs').glob('*/server.log')):
   item['log_progress_timestamp']=max(item['log_progress_timestamp'],p.stat().st_mtime)
  data['experiments'][folder.name]=item
 return data

def eligible_cards(job,snapshot,now):
 cards=[];local=now.astimezone(ZoneInfo('America/Chicago'));inside=(1,0)<=(local.hour,local.minute)<(9,30)
 end=local.replace(hour=9,minute=30,second=0,microsecond=0)
 for host,rows in snapshot.get('hosts',{}).items():
  if host not in job['hosts']:continue
  for index,row in rows.items():
   priority=host=='gpu01' and str(index) in ['1','2'];fresh=0<=now.timestamp()-row.get('last',0)<=25
   fits=priority or (inside and (end-now).total_seconds()>job['max_seconds']+120)
   if fits and fresh and row.get('qualified') is True and row.get('idle') is True and not row.get('processes') and row.get('utilization')==0 and row.get('used_mib',999999)<256:cards.append(f'{host}/{index}')
 return cards

def assess(data,config,now=None):
 now=now or utcnow();issues=[];pending=[];running=[];failed=[];complete=[];resolved=[]
 def issue(code,subject,detail,needs_agent=True):issues.append(dict(code=code,subject=subject,detail=detail,needs_agent=needs_agent))
 previous=data.get('last_takeover',{})
 if previous.get('status') in ['failed','timed_out'] or (previous.get('response') or {}).get('status')=='failed':issue('diagnostic_takeover_failed','Codex diagnostic',dict(status=previous.get('status'),error=previous.get('error'),run_dir=previous.get('run_dir')))
 if previous.get('status')=='running' and previous.get('started') and now.timestamp()-stamp(previous['started'])>config.get('agent_timeout_seconds',600)+60:issue('diagnostic_takeover_abandoned','Codex diagnostic',dict(started=previous['started'],run_dir=previous.get('run_dir')))
 if data.get('read_errors'):issue('unreadable_state','project',data['read_errors'])
 if data['snapshot'].get('errors'):issue('inventory_errors','GPU dispatcher',data['snapshot']['errors'])
 count=sum(len(rows) for rows in data['snapshot'].get('hosts',{}).values())
 if data.get('expected_gpu_count') and count!=data['expected_gpu_count']:issue('incomplete_gpu_inventory','16-GPU inventory',dict(expected=data['expected_gpu_count'],observed=count))
 fresh=data.get('inventory_age_seconds');controller_up=data['controller'].get('ActiveState')=='active'
 if not controller_up:issue('controller_down','openwam-nightly-gpu.service',data['controller'],False)
 if fresh is None or fresh>90 or fresh< -15:issue('inventory_stale','16-GPU inventory',fresh,False)
 for j in data['jobs']:
  claim=j.get('claim')
  if claim is None:
   pending.append(j['id'])
   if not j['authorization_valid']:issue('invalid_job_manifest',j['id'],'Registered authorization hash mismatch')
   if j['paused_by']:continue
   if fresh is not None and 0<=fresh<=25 and now.timestamp()-j['created_timestamp']>=120:
    cards=eligible_cards(j,data['snapshot'],now)
    if cards:issue('eligible_gpu_not_dispatched',j['id'],cards)
   # Deadlines are checked even while the job is waiting for GPU capacity.
   science=j.get('experiment') or j['scientific_identity'].split('/')[0];p=data['experiments'].get(science,{}).get('files',{}).get('protocol.json',{})
   if p.get('deadline') and now.timestamp()>=stamp(p['deadline']):issue('queued_job_expired',j['id'],p['deadline'])
  elif claim['status'] in TERMINAL:
   (failed if claim['status']=='failed' else complete).append(j['id'])
  else:
   running.append(j['id']);science=j.get('experiment') or j['scientific_identity'].split('/')[0]
   experiment=data['experiments'].get(science,{})
   # Native ex8-a's historical scientific_identity predates the directory prefix.
   if not experiment:experiment=data['experiments'].get(config.get('job_experiments',{}).get(j['id'],''),{})
   last=j.get('last_log_progress_timestamp',0);created=stamp(claim['created'])
   if now.timestamp()-max(last,created)>config['stall_seconds']:issue('experiment_no_log_progress',j['id'],dict(last_log=last,claimed=claim['created']))
   if claim['status']=='submission_uncertain':issue('uncertain_submission',j['id'],'Inspect owned unit/result; never submit the same job again')
 # A failed old attempt with a registered recovery is retained as history.
 known={j['id']:j for j in data['jobs']}
 for jid in failed:
  recovery=config.get('recoveries',{}).get(jid)
  if recovery in known:
   c=known[recovery].get('claim')
   if c is None or c['status']!='failed':resolved.append(dict(failed=jid,recovery=recovery));continue
  issue('failed_experiment',jid,known[jid]['claim'].get('worker_result',{}).get('error') or known[jid]['claim'].get('error') or 'Read preserved native logs for the cause')
 full=next((e['files'][rel] for e in data['experiments'].values() for rel in ['full/readout.json','readout.json'] if isinstance(e['files'].get(rel),dict) and e['files'][rel].get('total_task_mode_rows')==100),None)
 full_registered=any(e['files'].get('protocol.json',{}).get('full_registered_for_dispatch') for e in data['experiments'].values())
 done=bool(full and full.get('full_endpoint_complete') and full.get('practical_agreement_passed'))
 if full and full.get('full_endpoint_complete') and not full.get('practical_agreement_passed'):issue('reproduction_gap','official full suite',full.get('gaps_pp'))
 global_pause=pathlib.Path(config['state_dir'])/'paused.json'
 if not pending and not running and not done:issue('empty_queue_before_reproduction','official reproduction','No active or queued official work; diagnose and register the next bounded stage')
 for name,experiment in data['experiments'].items():
  stage=experiment['files'].get('stage-state.json',{})
  if stage.get('status','').startswith('hold_') or stage.get('status')=='gate_failed':issue('full_scaling_gate_hold',name,stage)
  report=experiment['files'].get('readout.json',{})
  if report.get('status')=='readiness_complete_full_not_started' and not full_registered and not running and not any(j.get('stage')=='native_full' and j['id'] in pending for j in data['jobs']):issue('readiness_needs_full_stage',name,{k:report.get(k) for k in ['next_gate','full_planning_estimate_gpu_hours','registered_full_ceiling_gpu_hours']})
 status='paused' if global_pause.exists() else 'needs_action' if issues else 'complete' if done else 'running' if running else 'waiting_for_gpu' if pending else 'unknown'
 return dict(status=status,issues=issues,pending_jobs=pending,running_jobs=running,failed_jobs=failed,complete_jobs=complete,addressed_failures=resolved,full_endpoint_complete=bool(full and full.get('full_endpoint_complete')),practical_agreement_passed=done,full_paper_reproduced=False)

def alert(state,event):
 with (state/'alerts.jsonl').open('a') as f:f.write(json.dumps(dict(time=utcnow().isoformat(),**event),ensure_ascii=False)+'\n')

def repair_controller(data,assessment,state):
 problems={i['code'] for i in assessment['issues']}
 if problems&{'controller_down','inventory_stale'}:
  action='restart' if data['controller'].get('ActiveState')=='active' else 'start'
  result=command(['systemctl','--user',action,'openwam-nightly-gpu.service'],timeout=30)
  alert(state,dict(type='controller_recovery',action=action,result=result));return [dict(action=action,unit='openwam-nightly-gpu.service',**result)]
 return []

def stop_group(child):
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:child.wait(timeout=10)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait(timeout=10)

def valid_user_hold(previous):
 response=previous.get('response') or {}
 return response.get('status')=='needs_user' and response.get('decision_basis') in ['explicit_user_constraint','automatic_approval_rejection','missing_credentials','external_dependency'] and bool(response.get('decision_evidence'))

def issue_fingerprint(assessment):
 body=[{k:i[k] for k in ['code','subject','detail']} for i in assessment['issues'] if i['needs_agent']]
 return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()

def run_agent(config,state,assessment,now):
 out=state/'runs'/now.strftime('%Y%m%dT%H%M%S.%fZ');out.mkdir(parents=True)
 prompt=(SOURCE/'AGENT_TASK.txt').read_text().replace('{{STATE}}',str(state)).replace('{{REPORT}}',str(state/'latest.json'))
 argv=[config['codex_binary'],'exec','--approve-for-me','--ephemeral','-C',config['worktree'],'--add-dir','/home/zifanz4','--json','--output-schema',str(SOURCE/'agent-response.schema.json'),'-o',str(out/'response.json'),'-']
 (out/'prompt.txt').write_text(prompt);write(out/'command.json',dict(argv=argv,timeout_seconds=config['agent_timeout_seconds']))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 # A fresh background invocation must not inherit the live interactive thread id.
 for key in ['CODEX_THREAD_ID','CODEX_SESSION_ID']:env.pop(key,None)
 start=time.monotonic();result=dict(started=now.isoformat(),status='running',run_dir=str(out),issue_fingerprint=issue_fingerprint(assessment));write(state/'agent.json',result)
 with (out/'events.jsonl').open('x') as log,(out/'stderr.log').open('x') as err:
  child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=log,stderr=err,env=env,start_new_session=True,text=True)
  try:child.communicate(input=prompt,timeout=config['agent_timeout_seconds'])
  except subprocess.TimeoutExpired:stop_group(child);result.update(status='timed_out',error='Bounded Codex takeover exceeded timeout; next timer tick still runs')
  except BaseException:stop_group(child);raise
  else:
   try:response=read(out/'response.json')
   except (ValueError,OSError) as exc:response=None;result['error']='Invalid or missing agent JSON: '+repr(exc)
   result.update(status='complete' if child.returncode==0 and response else 'failed',exit_code=child.returncode,response=response)
   events=(out/'events.jsonl').read_text(errors='replace')
   result['shell_command_events']=sum(1 for line in events.splitlines() if '"type":"command_execution"' in line or '"type": "command_execution"' in line)
 result.update(finished=utcnow().isoformat(),seconds=time.monotonic()-start);write(state/'agent.json',result);alert(state,dict(type='agent_finished',result=result));return result

def tick(config,state,check_only=False,force_agent=False):
 now=utcnow();data=collect(config,now);assessment=assess(data,config,now);paused=assessment['status']=='paused';actions=[]
 report=dict(checked_at=now.isoformat(),next_check_within_seconds=900,assessment=assessment,data=data,actions=actions,agent=None)
 write(state/'latest.json',report)
 if not check_only and not paused:
  actions=repair_controller(data,assessment,state);report['actions']=actions
  if actions:
   # Recheck on the next quarter-hour if the restarted inventory is still warming.
   data=collect(config);assessment=assess(data,config);report.update(data=data,assessment=assessment);write(state/'latest.json',report)
  actionable=any(i['needs_agent'] for i in assessment['issues'])
  last=read(state/'agent.json',{});previous=stamp(last['started']) if last.get('started') else 0
  due=now.timestamp()-previous>=config['agent_min_interval_seconds']
  requires_user=actionable and valid_user_hold(last) and last.get('issue_fingerprint')==issue_fingerprint(assessment)
  if requires_user and not force_agent:report['agent']='awaiting_required_user_input_checks_continue'
  elif force_agent or (actionable and due):
   alert(state,dict(type='takeover_requested',issues=assessment['issues']));report['agent']=run_agent(config,state,assessment,now)
   # Trust observable queue/experiment evidence, not just an agent's prose.
   after=collect(config);report.update(data=after,assessment=assess(after,config),checked_after_agent=utcnow().isoformat())
  elif actionable:report['agent']='cooldown_until_next_tick'
 write(state/'latest.json',report)
 summary=dict(time=utcnow().isoformat(),status=report['assessment']['status'],issues=[i['code'] for i in report['assessment']['issues']],pending=report['assessment']['pending_jobs'],running=report['assessment']['running_jobs'],actions=actions,agent_status=report['agent'].get('status') if isinstance(report['agent'],dict) else report['agent'],agent_outcome=(report['agent'].get('response') or {}).get('status') if isinstance(report['agent'],dict) else None)
 with (state/'history.jsonl').open('a') as f:f.write(json.dumps(summary,ensure_ascii=False)+'\n')
 write(state/'heartbeat.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(SOURCE/'config.json'));p.add_argument('--state-dir');p.add_argument('--check-only',action='store_true');p.add_argument('--force-agent',action='store_true');args=p.parse_args()
 config=read(args.config);state=pathlib.Path(args.state_dir or config['state_dir']);config['state_dir']=str(state);state.mkdir(parents=True,exist_ok=True)
 with (state/'bot.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:
   # Manual checks or an unusually timed takeover must not skip the next
   # quarter-hour observation. Observe, but never start a second agent.
   now=utcnow();data=collect(config,now);assessment=assess(data,config,now)
   report=dict(checked_at=now.isoformat(),assessment=assessment,data=data,actions=[],agent=read(state/'agent.json',{}),concurrent_takeover_observation=True)
   write(state/'latest.json',report)
   summary=dict(time=now.isoformat(),status=assessment['status'],issues=[i['code'] for i in assessment['issues']],pending=assessment['pending_jobs'],running=assessment['running_jobs'],agent_status='already_active_observed_without_duplicate')
   write(state/'heartbeat.json',summary)
   with (state/'history.jsonl').open('a') as f:f.write(json.dumps(summary,ensure_ascii=False)+'\n')
   print(json.dumps(summary,ensure_ascii=False),flush=True);return
  try:tick(config,state,args.check_only,args.force_agent)
  except BaseException as e:
   alert(state,dict(type='selfcheck_failed',error=repr(e),traceback=traceback.format_exc()[-10000:]));write(state/'heartbeat.json',dict(time=utcnow().isoformat(),status='bot_error',error=repr(e)));raise
if __name__=='__main__':main()
