#!/usr/bin/env python3
"""Execute upstream reporting logic with a stub rollout to expose denominator mismatch."""
import argparse,ast,contextlib,datetime,difflib,importlib.util,io,json,os,tempfile,types
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--robotwin',type=Path,default=ROOT/'benchmarks/RoboTwin');parser.add_argument('--openwam',type=Path,default=ROOT/'OpenWAM');args=parser.parse_args()
RT=args.robotwin
WRAP=args.openwam/'benchmarks/robotwin/eval_policy_wrapper.py'
src=(RT/'script/eval_policy.py').read_text()
fixed=src.replace('    test_num = 100\n','    test_num = int(os.environ.get("ROBOTWIN_TEST_NUM", "").strip() or "100")\n    if test_num <= 0:\n        raise ValueError("ROBOTWIN_TEST_NUM must be > 0")\n',1)
assert fixed!=src
spec=importlib.util.spec_from_file_location('official_wrapper',WRAP);wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)

def run(source,cap):
 tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 calls=[]
 def fake_eval(*args,**kwargs):
  count=kwargs['test_num'];calls.append(count);return args[4]+count,count # all attempted episodes succeed
 ns={'datetime':datetime.datetime,'Path':Path,'os':os,'np':np,'yaml':yaml,'CONFIGS_PATH':str(RT/'task_config')+'/',
     'eval_function_decorator':lambda *a:lambda cfg:None,'class_decorator':lambda *a:None,
     'get_embodiment_config':lambda *a:{'arm_joints_name':[['l'],['r']]},'get_camera_config':lambda *a:{'h':64,'w':64},'eval_policy':fake_eval}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(RT/'script/eval_policy.py'),'exec'),ns)
 module=types.SimpleNamespace(eval_policy=fake_eval)
 previous=os.environ.get('ROBOTWIN_TEST_NUM')
 if cap is None:os.environ.pop('ROBOTWIN_TEST_NUM',None)
 else:os.environ['ROBOTWIN_TEST_NUM']=str(cap)
 try:
  wrapper._install_test_num_override(module);ns['eval_policy']=module.eval_policy
  old=os.getcwd()
  with tempfile.TemporaryDirectory() as d:
   Path(d,'task_config').symlink_to(RT/'task_config',target_is_directory=True);os.chdir(d)
   try:
    with contextlib.redirect_stdout(io.StringIO()):ns['main']({'task_name':'pick_dual_bottles','task_config':'demo_clean','ckpt_setting':'stub','policy_name':'stub','instruction_type':'unseen','seed':0})
    files=list(Path(d).rglob('_result.txt'));assert len(files)==1
    rate=float(files[0].read_text().strip().splitlines()[-1])
   finally:os.chdir(old)
  return {'rollouts_requested':calls,'all_rollouts_succeeded':True,'written_success_rate':rate}
 finally:
  if previous is None:os.environ.pop('ROBOTWIN_TEST_NUM',None)
  else:os.environ['ROBOTWIN_TEST_NUM']=previous

result={'original_capped':run(src,5),'fixed_capped':run(fixed,5),'fixed_default':run(fixed,None),'fixed_empty':run(fixed,'')}
assert result['original_capped']['rollouts_requested']==[5] and result['original_capped']['written_success_rate']==.05
assert result['fixed_capped']['rollouts_requested']==[5] and result['fixed_capped']['written_success_rate']==1.
assert result['fixed_default']['rollouts_requested']==[100] and result['fixed_default']['written_success_rate']==1.
assert result['fixed_empty']['rollouts_requested']==[100] and result['fixed_empty']['written_success_rate']==1.
(ROOT/'contribution').mkdir(exist_ok=True)
(ROOT/'contribution/robotwin-smoke-denominator.patch').write_text(''.join(difflib.unified_diff(src.splitlines(True),fixed.splitlines(True),fromfile='a/script/eval_policy.py',tofile='b/script/eval_policy.py')))
(ROOT/'contribution/robotwin-smoke-denominator-reproduction.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
