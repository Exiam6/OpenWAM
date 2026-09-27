#!/usr/bin/env python3
"""CPU check of the exact upstream rollout loop after the replay transformation."""
import ast,contextlib,io,json,types
from pathlib import Path
import numpy as np
from replay_confirmation_client import install_replay
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/robustness-20260921'
source=ROOT/'benchmarks/RoboTwin/script/eval_policy.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='eval_policy')
module=types.ModuleType('replay_checked');module.np=np
seen=[]
class Env:
 def setup_demo(self,*,seed,**kw):
  self.seed=seed;self.take_action_cnt=0;self.step_lim=1;self.eval_success=False;self.eval_video_path=None;self.render_freq=0
 def play_once(self):raise AssertionError('Expert must not be rerun during manifest replay')
 def set_instruction(self,*,instruction):self.instruction=instruction
 def get_obs(self):return {}
 def close_env(self,**kw):seen.append({'seed':self.seed,'instruction':self.instruction})
module.class_decorator=lambda task:Env()
module.generate_episode_descriptions=lambda *a:None
scenes=json.loads((OUT/'fixed-scenes.json').read_text())['scenes']
def act(env,model,obs):env.take_action_cnt+=1;env.eval_success=True
module.eval_function_decorator=lambda name,kind:act if kind=='eval' else lambda model:None
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),module.__dict__)
install_replay(module,scenes)
env=module.class_decorator('pick_dual_bottles');args={'task_name':'pick_dual_bottles','policy_name':'fake','render_freq':0,'clear_cache_freq':5,'task_config':'demo_clean','ckpt_setting':'check'}
with contextlib.redirect_stdout(io.StringIO()):
 end,success=module.eval_policy('pick_dual_bottles',env,args,None,300000,test_num=5,instruction_type='unseen')
assert success==5 and [r['seed']for r in seen]==[r['seed']for r in scenes]
assert [r['instruction']for r in seen]==[r['instruction']for r in scenes]
np.random.seed(42);np.random.choice(['a','b','c','d','e']);expected=np.random.get_state()
np.random.seed(42);np.random.choice(['fixed']*5);actual=np.random.get_state()
assert all(np.array_equal(a,b)for a,b in zip(expected,actual))
result={'exact_upstream_rollout_loop_executed':True,'fixed_seed_order_and_prompts':True,'expert_not_refiltered':True,'choice_rng_consumption_preserved':True,'fake_success_count':success,'scenes':seen}
(OUT/'fixed-scene-checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
