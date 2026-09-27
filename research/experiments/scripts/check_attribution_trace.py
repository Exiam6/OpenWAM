"""CPU gate using native eval and the real upstream three-scene loop."""
import ast,contextlib,copy,io,json,random,sys,tempfile,types
from pathlib import Path
import numpy as np
from repeat_common import payload_hash
from capture_repeat_inputs import make_recording_model,capture_payload
from attribution_trace import AttributionTrace,FixedCommandTransport,snapshot
from trajectory_trace import instrument_interface
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/attribution-20260921'
sys.path.insert(0,str(ROOT/'OpenWAM/benchmarks/robotwin'))
import openwam2robotwin_interface as interface

def main():
    class Pose:
        def __init__(self):self.p=np.zeros(3,np.float32);self.q=np.array([1,0,0,0],np.float32)
    class Entity:
        def __init__(self):self.qpos=np.array([.123,0,0],np.float32);self.qvel=np.zeros(3,np.float32);self.global_pose=Pose()
        def get_qpos(self):return self.qpos
        def get_qvel(self):return self.qvel
        def get_root_pose(self):return self.global_pose
        def get_pose(self):return self.global_pose
    class Planner:
        def plan_path(self,curr_joint_pos,target_gripper_pose,constraint_pose=None,arms_tag=None):
            self.seen=(curr_joint_pos,target_gripper_pose);self.result={'status':'Success','position':np.array([curr_joint_pos,curr_joint_pos+.1]),'velocity':np.zeros((2,3),np.float32)};return self.result
    class Env:
        task_name='pick_dual_bottles'
        def __init__(self):
            self.take_action_cnt=0;self.step_lim=400;self.eval_success=False
            self.robot=types.SimpleNamespace(left_entity=Entity(),right_entity=Entity(),left_ee=Entity(),right_ee=Entity(),left_planner=Planner(),right_planner=Planner(),communication_flag=False,get_left_arm_jointState=lambda:[0.,0.,0.],get_right_arm_jointState=lambda:[0.,0.,0.])
            self.bottle1=Entity();self.bottle2=Entity();self.received=[]
        def get_instruction(self):return 'fixed instruction'
        def take_action(self,a,action_type):
            self.received.append((a,action_type))
            for side in ['left','right']:
                planner=getattr(self.robot,side+'_planner');entity=getattr(self.robot,side+'_entity');q=entity.get_qpos();pose=Pose()
                result=planner.plan_path(q,pose,arms_tag=side);assert planner.seen[0]is q and planner.seen[1]is pose and result is planner.result
                entity.qpos=result['position'][-1].copy()
            self.take_action_cnt+=1;return 'native-return'
    ob={'observation':{k:{'rgb':np.full((8,9,3),i,np.uint8)}for i,k in enumerate(['head_camera','left_camera','right_camera'])},'endpose':{'left_endpose':[0,0,0,0,0,0,1],'right_endpose':[1,2,3,0,0,0,1],'left_gripper':.25,'right_gripper':.75},'joint_action':{'vector':np.zeros(6)}}
    recorder=make_recording_model(interface);payload=capture_payload(interface,recorder,ob,'fixed instruction')
    a=interface._extract_eef_proprio(ob).tolist();commands={'400000':[{'server_action':a},{'server_action':a}]}
    interface._STEP_LIM_OVERRIDES={**interface._STEP_LIM_OVERRIDES,'pick_dual_bottles':2}
    def native_model(trace):
        m=make_recording_model(interface);m._client=FixedCommandTransport(trace,commands);return m
    base=Env();base_trace=types.SimpleNamespace(current=None);bm=native_model(base_trace)
    for i in range(2):base_trace.current={'seed':400000,'step':i};interface.eval(base,bm,ob)
    with tempfile.TemporaryDirectory()as td:
        trace=AttributionTrace(td,{('clean',400000):payload_hash(payload)},interface._extract_eef_proprio);env=Env();trace.env=env
        original_left=env.robot.left_planner.plan_path;model=native_model(trace)
        proxy=types.SimpleNamespace(get_model=lambda args:model,eval=interface.eval);instrument_interface(proxy,trace);m=proxy.get_model({})
        before_np=np.random.get_state();before_py=random.getstate()
        for i in range(2):trace.begin(ob,ob,'clean',400000,i);proxy.eval(env,m,ob)
        assert env.take_action_cnt==base.take_action_cnt==2 and env.step_lim==2
        assert all(np.array_equal(x[0],y[0])and x[1]==y[1]=='ee'for x,y in zip(base.received,env.received))
        assert snapshot(env)==snapshot(base) and env.robot.left_planner.plan_path==original_left
        assert all(np.array_equal(a,b)for a,b in zip(before_np,np.random.get_state()))and before_py==random.getstate()
        rows=[json.loads(x)for x in(Path(td)/'seed-400000-trace.jsonl').read_text().splitlines()]
        assert len(rows)==2 and all(len(r['planner_calls'])==2 for r in rows)
        assert rows[0]['before_execution']['actual_articulations']['left']['qpos']['values'][0]!=0
        assert rows[0]['before_execution']['joint_drive_targets']['values']==[0.]*6
        assert rows[0]['planner_calls'][0]['result']['position']['values']!=rows[1]['planner_calls'][0]['result']['position']['values']
        trace.begin(ob,ob,'clean',400000,2)
        try:FixedCommandTransport(trace,commands).predict(payload)
        except AssertionError:pass
        else:raise AssertionError('third command was not rejected')
    from norm_client import install_replay
    source=ROOT/'benchmarks/RoboTwin/script/eval_policy.py';node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='eval_policy')
    module=types.ModuleType('attribution_loop_test');module.np=np;seen=[];rng=[]
    class LoopEnv:
        task_name='pick_dual_bottles'
        def setup_demo(self,*,seed,**kw):self.seed=seed;np.random.seed(seed);self.take_action_cnt=0;self.step_lim=400;self.eval_success=False;self.eval_video_path=None;self.render_freq=0
        def play_once(self):raise AssertionError('refilter')
        def set_instruction(self,*,instruction):self.instruction=instruction
        def get_obs(self):return {}
        def close_env(self,**kw):seen.append((self.seed,self.take_action_cnt,self.instruction))
    def act(env,model,ob):
        if env.take_action_cnt==0:rng.append(np.random.get_state())
        interface._apply_step_lim_override(env);env.take_action_cnt+=1
    module.class_decorator=lambda task:LoopEnv();module.generate_episode_descriptions=lambda*a:None;module.eval_function_decorator=lambda name,kind:act if kind=='eval'else lambda model:None
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),module.__dict__)
    scenes=[{'seed':400000+i,'instruction':f'prompt{i}','instruction_choice_length':10}for i in range(3)];install_replay(module,scenes)
    args={'task_name':'pick_dual_bottles','policy_name':'fake','render_freq':0,'clear_cache_freq':5,'task_config':'demo_clean','ckpt_setting':'two-command-probe'}
    with contextlib.redirect_stdout(io.StringIO()):module.eval_policy('pick_dual_bottles',module.class_decorator('task'),args,None,1,test_num=3,instruction_type='unseen')
    assert [x[1]for x in seen]==[2,2,2]
    for i,(seed,count,prompt)in enumerate(seen):
        assert seed==400000+i and prompt==f'prompt{i}';np.random.seed(seed);np.random.choice(range(10));assert all(np.array_equal(a,b)for a,b in zip(rng[i],np.random.get_state()))
    checks={k:True for k in ['native_eval_equal_actions_types_and_physics','planner_argument_and_return_identity','planner_methods_restored','rng_unchanged','actual_qpos_distinct_from_drive_targets','planner_arrays_copied_before_mutation','third_command_rejected','native_three_scene_loop_exactly_two_actions','fixed_prompts_and_ten_choice_rng']}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cpu-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
