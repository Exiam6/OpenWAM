"""Execute exact native grasp_actor AST without a simulator or a task rollout."""
import ast,datetime,hashlib,json,sys
from pathlib import Path
base=Path('/data02/zifanz4/openwam-experiments/benchmarks/RoboTwin');out=Path('/home/zifanz4/openwam-experiments/studies/goalaux-20260922/null-grasp-check.json')
tree=ast.parse((base/'envs/_base_task.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Base_Task');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='grasp_actor');method.decorator_list=[]
# Strip annotations only; the native body is unchanged.
method.returns=None
for a in method.args.args: a.annotation=None
module=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(module)
action_tree=ast.parse((base/'envs/utils/action.py').read_text());action=next(n for n in action_tree.body if isinstance(n,ast.ClassDef) and n.name=='Action');move_assert=next(n for n in ast.walk(action) if isinstance(n,ast.Assert) and isinstance(n.msg,ast.Constant) and n.msg.value=='target_pose cannot be None for move action.')
checks=[]
def Action(arm,kind,target_pose=None,**kwargs):
 if kind=='move':
  exec(compile(ast.Module(body=[move_assert],type_ignores=[]),'native-action-assert','exec'),{},dict(target_pose=target_pose))
 return (arm,kind,target_pose,kwargs)
ns={'Action':Action};exec(compile(module,'native-grasp-actor','exec'),ns)
class Fixture:
 plan_success=True;need_plan=True
 def __init__(self,pair):self.pair=pair
 def choose_grasp_pose(self,*args,**kwargs):return self.pair
f=ns['grasp_actor']
try:f(Fixture((None,None)),object(),'left')
except AssertionError as exc:assert str(exc)=='target_pose cannot be None for move action.';checks.append('exact_native_null_sentinel_reaches_assertion')
else:raise AssertionError('Expected native null failure')
a=[0,0,0,1,0,0,0];b=[.1,0,0,1,0,0,0];assert len(f(Fixture((a,a)),object(),'left')[1])==2 and len(f(Fixture((a,b)),object(),'right')[1])==3;checks.append('valid_native_branches_unchanged')
x=Fixture((None,None));x.plan_success=False;assert f(x,object(),'left')==(None,[]);checks.append('native_plan_failure_short_circuit')
from null_grasp_classification import is_native_no_grasp
r=json.loads(Path(sys.argv[1]).read_text());assert is_native_no_grasp(r)
for update in [{'error':"AssertionError('other')"},{'task':'handover_block'},{'traceback':r['traceback'].replace('line 73','line 125')},{'outcome':'expert_infeasible'},{'exit_code':124},{'accepted':True}]:assert not is_native_no_grasp(dict(r,**update))
checks.append('classifier_rejects_other_errors_tasks_phases_timeouts_acceptances')
files=['envs/_base_task.py','envs/place_object_basket.py','envs/utils/action.py'];d={'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),'checks':checks,'no_simulation_or_completed_scene_replay':True,'no_performance_scores_observed':True,'native_source_sha256':{f:hashlib.sha256((base/f).read_bytes()).hexdigest() for f in files}};out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
