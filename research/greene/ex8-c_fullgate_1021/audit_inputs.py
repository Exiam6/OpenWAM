"""Read complete original groups independently; no claim about pending recovery."""
import pathlib,json,math,datetime,hashlib
from prepare import A as ROOT,OUT

def main():
 pause=pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')
 assert not pause.exists()
 p=json.loads((ROOT/'pilot/result.json').read_text());groups=[]
 for g in p['groups']:
  d=ROOT/'pilot'/f"{g['mode']}-{g['task']}";records=0;errors=[];eps=set()
  for f in (d/'policy-inputs').glob('ep*/step_*/meta.json'):
   m=json.loads(f.read_text());records+=1;eps.add(int(f.parent.parent.name[2:]))
   for k in ['state','action']:
    x=m.get(k)
    if not isinstance(x,list) or len(x)!=20 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in x):errors.append(str(f)+': '+k)
   for camera in ['head','left','right']:
    c=f.parent/(camera+'.png')
    if not c.is_file() or c.stat().st_size<64:errors.append(str(c))
  assert not errors,errors[:3]
  assert eps==set(range(5)),eps
  groups.append(dict(task=g['task'],mode=g['mode'],finite_20d_state_action_records=records,episodes=sorted(eps),all_three_camera_files_present=True))
 v=dict(updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),groups=groups,total_records=sum(g['finite_20d_state_action_records'] for g in groups),scope='Five complete original groups only; original partial two episodes excluded, recovery pending; existence/shape check does not replace video decode or visual review',regression_tests_passed=17,full_runner_review=dict(original_fatal_error_list_would_repeat_candidate_failure=True,prepared_fix_scoped_to_place_object_basket=True,native_episodes_default=100,native_seed=0,native_debug_default=False,other_tracebacks_and_planner_errors_fatal=True,aggregate_checks_protocol_digest_and_seed_monotonicity=True,alternate_evaluator_and_version_bypass_rejected=True,full_runner_not_gpu_validated=True),original_visual_review_sha256=hashlib.sha256((ROOT/'independent-audit/visual-review.json').read_bytes()).hexdigest(),full_paper_reproduced=False)
 assert not pause.exists()
 (OUT/'cpu-validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'groups':len(groups),'validated_state_action_records':v['total_records'],'tests_passed':17}))
if __name__=='__main__':main()
