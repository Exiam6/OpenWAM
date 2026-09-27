"""Exercise the real native interface before launching passive trajectory tracing."""
import ast,contextlib,gzip,io,json,random,sys,tempfile,types
from pathlib import Path
import numpy as np
import yaml
from repeat_common import payload_hash
from capture_repeat_inputs import make_recording_model,capture_payload
from trajectory_trace import TrajectoryTrace,instrument_interface
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/repeatability-20260921/closedloop'
sys.path.insert(0,str(ROOT/'OpenWAM/benchmarks/robotwin'))
import openwam2robotwin_interface as interface

def main():
    task_limit=yaml.safe_load((ROOT/'benchmarks/RoboTwin/task_config/_eval_step_limit.yml').read_text())['pick_dual_bottles']
    assert task_limit==400 and 'pick_dual_bottles' not in interface._STEP_LIM_OVERRIDES
    observation={'observation':{k:{'rgb':np.full((8,9,3),i,dtype=np.uint8)}for i,k in enumerate(['head_camera','left_camera','right_camera'])},'endpose':{'left_endpose':[0,0,0,0,0,0,1],'right_endpose':[1,2,3,0,0,0,1],'left_gripper':.25,'right_gripper':.75},'joint_action':{'vector':np.linspace(0,1e-9,14,dtype=np.float64)}}
    recorder=make_recording_model(interface);payload=capture_payload(interface,recorder,observation,'fixed instruction')
    action=interface._extract_eef_proprio(observation)
    response={'action':action.tolist(),'step':1}
    class Transport:
        def reset(self):return {'type':'reset_ack'}
        def predict(self,p):self.received=p;return response
    def model():
        m=make_recording_model(interface);m._client=Transport();return m
    class Env:
        task_name='pick_dual_bottles';take_action_cnt=0;eval_success=False;step_lim=task_limit
        def get_instruction(self):return 'fixed instruction'
        def take_action(self,a,*args,**kw):self.received=a;self.kw=kw;self.take_action_cnt+=1;return 'sentinel'
    # Prime lazy imports before measuring the passive wrappers' RNG behavior.
    base=Env();base_model=model();interface.eval(base,base_model,observation)
    assert base.step_lim==400 and base.take_action_cnt==1
    with tempfile.TemporaryDirectory()as td:
        trace=TrajectoryTrace(td,{('clean',400000):payload_hash(payload)},interface._extract_eef_proprio)
        m=model();trace.begin(observation,observation,'clean',400000,0)
        proxy=types.SimpleNamespace(get_model=lambda args:m,eval=interface.eval)
        instrument_interface(proxy,trace);wrapped=proxy.get_model({});assert wrapped is m
        np_before=np.random.get_state();py_before=random.getstate()
        env=Env();proxy.eval(env,wrapped,observation)
        assert np.array_equal(env.received,base.received)and env.kw==base.kw and env.step_lim==400
        assert all(np.array_equal(a,b)for a,b in zip(np_before,np.random.get_state())) and py_before==random.getstate()
        row=json.loads((Path(td)/'seed-400000-trace.jsonl').read_text())
        assert np.array_equal(np.array(row['env_action']),env.received)
        assert row['joint_dtype']=='float64' and np.array_equal(np.array(row['joint_state']),observation['joint_action']['vector'])
        decoded=json.loads(gzip.decompress((Path(td)/row['request_file']).read_bytes()))
        assert decoded==payload and payload_hash(decoded)==row['request_sha256']
        # Verify identity at both direct wrapper seams, including arbitrary returns.
        trace.begin(observation,observation,'clean',400000,0)
        trace.current['step']=1  # separate local test request filename/counter
        ret={'action':action.tolist(),'step':2};observed=[]
        assert trace.predict(lambda p:(observed.append(p)or ret),payload)is ret and observed[0]is payload
        value=env.received;received=[];sentinel=object()
        assert trace.take_action(lambda a,**kw:(received.append(a)or sentinel),value,action_type='ee')is sentinel
        assert received[0]is value
    # Execute the unchanged upstream loop transformed only by frozen manifest replay.
    from norm_client import install_replay
    source=ROOT/'benchmarks/RoboTwin/script/eval_policy.py'
    tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval_policy')
    module=types.ModuleType('stagec_loop_check');module.np=np;seen=[];rng_at_start=[]
    class LoopEnv:
        task_name='pick_dual_bottles'
        def setup_demo(self,*,seed,**kw):
            self.seed=seed;np.random.seed(seed);self.take_action_cnt=0;self.step_lim=task_limit;self.eval_success=False;self.eval_video_path=None;self.render_freq=0
        def play_once(self):raise AssertionError('no refilter')
        def set_instruction(self,*,instruction):self.instruction=instruction
        def get_obs(self):return {}
        def close_env(self,**kw):seen.append((self.seed,self.take_action_cnt,self.instruction))
    def act(env,model,obs):
        if env.take_action_cnt==0:rng_at_start.append(np.random.get_state())
        interface._apply_step_lim_override(env);env.take_action_cnt+=1
        env.eval_success=env.seed!=400002 and env.take_action_cnt==2
    module.class_decorator=lambda task:LoopEnv();module.generate_episode_descriptions=lambda *a:None
    module.eval_function_decorator=lambda name,kind:act if kind=='eval' else lambda model:None
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),module.__dict__)
    scenes=[{'seed':400000+i,'instruction':f'prompt{i}','instruction_choice_length':10}for i in range(3)]
    install_replay(module,scenes)
    args={'task_name':'pick_dual_bottles','policy_name':'fake','render_freq':0,'clear_cache_freq':5,'task_config':'demo_clean','ckpt_setting':'trace-check'}
    with contextlib.redirect_stdout(io.StringIO()):_,success=module.eval_policy('pick_dual_bottles',module.class_decorator('task'),args,None,1,test_num=3,instruction_type='unseen')
    assert success==2 and [x[1]for x in seen]==[2,2,400]
    for i,(seed,count,prompt)in enumerate(seen):
        assert seed==400000+i and prompt==f'prompt{i}'
        np.random.seed(seed);np.random.choice(list(range(10)))
        assert all(np.array_equal(a,b)for a,b in zip(rng_at_start[i],np.random.get_state()))
    checks={'actual_native_eval_equal_actions_and_types':True,'upstream_task_limit_400_preserved':True,'native_modelclient_used':True,'wrapper_input_and_return_object_identity':True,'rng_unchanged':True,'all_three_cameras_and_eef_joint_traced':True,'gzip_request_roundtrip_exact':True,'actual_three_scene_replay_and_ten_choice_rng':True,'upstream_loop_enforces400_action_cap':True}
    (OUT/'cpu-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
