"""CPU-only evidence and cost draft. Never writes GPU jobs or L4 authorization."""
import datetime as dt, hashlib, json, pathlib
from run_full import score, classify_full
R=pathlib.Path('/home/zifanz4/openwam-runtime')
A=R/'ex8-a_officialrepro_1021'; B=R/'ex8-b_candidatefilter_1021'
OUT=R/'ex8-c_fullgate_1021'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def allocation(base, cap):
 limits={}
 for line in (A/'RoboTwin/task_config/_eval_step_limit.yml').read_text().splitlines():
  if line.strip() and not line.lstrip().startswith('#'):
   k,v=line.split(':',1);limits[k.strip()]=int(v)
 weight=2*sum(limits.get(t,1000) for t in base['full_tasks']);pool=int(cap*3600)-600*100
 rows=[dict(key=f'{m}-{t}',task=t,mode=m,episodes=100,seed=0,native_step_limit=limits.get(t,1000),max_seconds=600+int(pool*limits.get(t,1000)/weight)) for m in base['full_modes'] for t in base['full_tasks']]
 assert len(rows)==len({r['key'] for r in rows})==100
 assert sum(r['max_seconds'] for r in rows)<=cap*3600
 return rows

def main():
 assert not pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
 assert not (A/'paused.json').exists() and not (B/'paused.json').exists()
 base=read(A/'protocol.json'); parent=read(A/'pilot/result.json'); recovery=read(B/'protocol.json');sync=read(R/'official-sync/latest.json')
 assert sync['passed'] and sync['after']==base['official_sync']['after']==recovery['official_sync']['after']
 assert recovery['parent_protocol_sha256']==sha(A/'protocol.json')
 assert recovery['parent_result_sha256']==sha(A/'pilot/result.json')
 assert recovery['checkpoint_sha256']==base['checkpoint_sha256']
 groups=parent['groups'];assert len(groups)==5 and all(g['status']=='complete' and g['episodes']==5 for g in groups)
 audited=[]
 for g in groups:
  folder=A/'pilot'/f"{g['mode']}-{g['task']}"; text=(folder/'client.log').read_text(); rows=score(text)
  assert len(rows)==5 and sum(x['success'] for x in rows)==g['successes']
  assert [x['scene_seed'] for x in rows]==g['accepted_scene_seeds']
  assert not classify_full(text,g['task'],True)['fatal']
  assert len(g['videos'])==5 and all(pathlib.Path(v).is_file() for v in g['videos'])
  audited.append(dict(task=g['task'],mode=g['mode'],scored_episodes=5,successes=g['successes'],client_log_sha256=sha(folder/'client.log')))
 seconds=sum(g['seconds'] for g in groups);lower=seconds/30*10000/3600+5
 missing_proxy=next(g['seconds'] for g in groups if g['task']=='place_object_basket');proxy=(seconds+missing_proxy)/30*10000/3600+5
 full=allocation(base,200); proposed=allocation(base,280)
 rr=B/'pilot/result.json'; r=read(rr) if rr.exists() else {'status':'not_started'}
 blockers=['L4 cost lower bound already exceeds authorized 200 GPUh','Separate preregistered immutable L4 manifest and dispatcher submission required']
 if r['status']!='pilot_complete':blockers.append('Recovery has not completed: '+r['status'])
 blockers.append('Independent combined action/count/video audit and recovery visual review required')
 draft=dict(kind='CPU_PREPARATION_NOT_GPU_AUTHORIZATION',updated=dt.datetime.now(dt.timezone.utc).isoformat(),official_sync=sync,compatibility='Same official commit and checkpoint hash as both frozen attempts; no upgrade or runtime edits',parent_status=parent['status'],original_failure_retained=True,original_partial_episodes_excluded=2,completed_groups=audited,recovery_status=r['status'],readiness_passed=False,full_endpoint_complete=False,full_paper_reproduced=False,known_group_seconds=seconds,formula='sum(six complete group seconds)/30*10000/3600 + 5 startup GPUh',full_cost_lower_bound_gpu_hours=lower,authorized_ceiling_gpu_hours=200,cost_gate_passed=False,proxy_full_gpu_hours=proxy,proxy_note='Unmeasured randomized basket group replaced only for planning by measured clean basket time; not a measured full-suite cost',proposed_ceiling_for_user_review_gpu_hours=280,proposal='Conditional 280 GPUh ceiling, only after final six-group estimate <=280 and all L0-L3 checks pass; 100 rows, 600 seconds startup per row, remainder proportional to native step limit; no deadline or GPU policy changes authorized',proposed_margin_over_proxy_gpu_hours=280-proxy,rows_at_existing_ceiling=full,proposed_rows_not_authorized=proposed,planned_episodes=10000,all_pilot_episodes_excluded_from_full_score=True,blockers=blockers,evidence_sha256={str(p):sha(p) for p in [A/'protocol.json',A/'pilot/result.json',B/'protocol.json',A/'pilot/demo_randomized-place_object_basket/client.log',pathlib.Path(__file__).parent/'run_full.py',pathlib.Path(__file__).parent/'classify_log.py']})
 OUT.mkdir(exist_ok=True)
 assert not pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
 (OUT/'preparation.json').write_text(json.dumps(draft,indent=2)+'\n')
 print(json.dumps({k:draft[k] for k in ['full_cost_lower_bound_gpu_hours','proxy_full_gpu_hours','recovery_status','cost_gate_passed','blockers']},indent=2))
if __name__=='__main__':main()
