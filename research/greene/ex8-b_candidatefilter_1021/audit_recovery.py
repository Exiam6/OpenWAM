"""CPU-only combined coverage audit. Does not schedule full evaluation."""
import pathlib,json,datetime as dt,time,math
ROOT=pathlib.Path('/home/zifanz4/openwam-runtime/ex8-b_candidatefilter_1021');BASE=ROOT.parent/'ex8-a_officialrepro_1021'
def read(p):return json.loads(p.read_text())
def write(v):
 t=ROOT/'readout.tmp';t.write_text(json.dumps(v,indent=2)+'\n');t.replace(ROOT/'readout.json')
while True:
 now=dt.datetime.now(dt.timezone.utc);f=ROOT/'pilot/result.json';r=read(f) if f.exists() else dict(status='waiting_for_eligible_GPU')
 out=dict(updated=now.isoformat(),status=r['status'],full_paper_reproduced=False,full_suite_started=False,parent_failure_retained=True,old_partial_episodes_excluded=2)
 if r['status']=='pilot_complete':
  assert r['readiness_passed'] and len(r['groups'])==1;g=r['groups'][0];assert g['mode']=='demo_randomized' and g['task']=='place_object_basket' and g['episodes']==5
  count=0
  for f in (ROOT/'pilot/demo_randomized-place_object_basket/policy-inputs').glob('ep*/step_*/meta.json'):
   m=read(f)
   for k in ['action','state']:assert len(m[k])==20 and all(math.isfinite(x) for x in m[k])
   count+=1
  groups=read(BASE/'pilot/result.json')['groups']+[g];assert len(groups)==6 and len({(x['task'],x['mode']) for x in groups})==6
  estimate=sum(x['seconds'] for x in groups)/30*10000/3600+5
  out.update(status='readiness_complete_full_not_started',groups=groups,scored_episodes=sum(x['episodes'] for x in groups),successes=sum(x['successes'] for x in groups),validated_recovery_action_records=count,full_planning_estimate_gpu_hours=estimate,registered_full_ceiling_gpu_hours=200,next_gate='full_cost_review' if estimate>200 else 'review_recovery_video_and_register_separate_L4_manifest')
 write(out)
 if r['status'] in ['pilot_complete','failed'] or now>dt.datetime.fromisoformat(read(ROOT/'protocol.json')['deadline']):break
 time.sleep(30)
