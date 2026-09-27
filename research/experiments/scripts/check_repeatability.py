"""Check real native replay and serialization seams without GPU inference."""
import ast,contextlib,io,json,random,sys,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/repeatability-20260921'
from norm_client import install_replay
from capture_repeat_inputs import make_recording_model,capture_payload
from repeat_common import traced_generate,difference,payload_hash,fresh_payload
sys.path.insert(0,str(ROOT/'OpenWAM/benchmarks/robotwin'))
import openwam2robotwin_interface as interface
from benchmarks.utils import client
from benchmarks.robotwin.prompt_template import format_prompt_for_inference

def main():
    source=ROOT/'benchmarks/RoboTwin/script/eval_policy.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='eval_policy')
    m=types.ModuleType('replay_check');m.np=np;seen=[]
    class Env:
        def setup_demo(self,*,seed,**kw):
            np.random.seed(seed);self.seed=seed;self.take_action_cnt=0;self.step_lim=400;self.eval_success=False;self.eval_video_path=None;self.render_freq=0
        def play_once(self):raise AssertionError('no refilter')
        def set_instruction(self,*,instruction):self.instruction=instruction
        def get_obs(self):return {'initial':True}
        def close_env(self,**kw):assert self.take_action_cnt==0
    def capture(env,model,obs):
        assert obs=={'initial':True};seen.append((env.seed,env.instruction,np.random.get_state()));env.step_lim=0
    m.class_decorator=lambda name:Env();m.generate_episode_descriptions=lambda *a:None
    m.eval_function_decorator=lambda name,kind:capture if kind=='eval' else lambda model:None
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),m.__dict__)
    scenes=[{'seed':400000+i,'instruction':f'prompt{i}','instruction_choice_length':10}for i in range(3)]
    install_replay(m,scenes)
    args={'task_name':'pick_dual_bottles','policy_name':'fake','render_freq':0,'clear_cache_freq':5,'task_config':'demo_clean','ckpt_setting':'capture'}
    with contextlib.redirect_stdout(io.StringIO()):_,success=m.eval_policy('pick_dual_bottles',m.class_decorator('task'),args,None,1,test_num=3,instruction_type='unseen')
    assert success==0 and len(seen)==3
    for (seed,prompt,state),row in zip(seen,scenes):
        assert (seed,prompt)==(row['seed'],row['instruction']);np.random.seed(seed);np.random.choice(list(range(10)))
        assert all(np.array_equal(a,b)for a,b in zip(state,np.random.get_state()))
    model=make_recording_model(interface)
    obs={'observation':{k:{'rgb':np.full((8,9,3),i,dtype=np.uint8)}for i,k in enumerate(['head_camera','left_camera','right_camera'])},'endpose':{'left_endpose':[0,0,0,0,0,0,1],'right_endpose':[1,2,3,0,0,0,1],'left_gripper':.25,'right_gripper':.75}}
    state=interface._extract_proprio(model,obs);rng=random.getstate();nprng=np.random.get_state()
    p=capture_payload(interface,model,obs,'fixed instruction')
    expected=client.build_payload(head=client.encode_numpy_b64(obs['observation']['head_camera']['rgb']),left_wrist=client.encode_numpy_b64(obs['observation']['left_camera']['rgb']),right_wrist=client.encode_numpy_b64(obs['observation']['right_camera']['rgb']),prompt=format_prompt_for_inference('fixed instruction'),state=[float(v)for v in state.astype(np.float32)])
    assert p==expected and payload_hash(p)==payload_hash(json.loads(json.dumps(p)))
    assert random.getstate()==rng and all(np.array_equal(a,b)for a,b in zip(nprng,np.random.get_state()))
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    pre=ObsPreprocessor(multiview=True,camera_layout=['head_camera','left_camera','right_camera'],img_height=384,img_width=320,requires_proprio=True)
    original_hash=payload_hash(p)
    first=pre.preprocess(fresh_payload(p));second=pre.preprocess(fresh_payload(p))
    assert first['image'].tobytes()==second['image'].tobytes() and np.array_equal(first['state'],second['state'])
    assert payload_hash(p)==original_hash and isinstance(p['state'],list) and 'image'not in p
    rng=random.getstate();nprng=np.random.get_state()
    actions=np.arange(40,dtype=np.float32).reshape(2,20);result={'actions':actions,'other':object()};sink=[]
    output=traced_generate(lambda value:result,sink)({'test':1});assert output is result and output['actions'] is actions
    assert len(sink)==1 and np.array_equal(sink[0],actions)and not np.shares_memory(sink[0],actions)
    assert random.getstate()==rng and all(np.array_equal(a,b)for a,b in zip(nprng,np.random.get_state()))
    assert difference(actions,actions)['byte_equal'];b=actions.copy();b[:,19]+=2
    diff=difference(b,actions);assert diff['max_abs']==2 and diff['groups']['xyz']['max_abs']==0 and diff['groups']['rot6d']['max_abs']==0
    out={'native_three_scene_zero_action_loop':True,'original_ten_way_instruction_rng':True,'native_modelclient_payload_exact':True,'payload_roundtrip_hash_exact':True,'trace_preserves_return_object_and_actions':True,'trace_and_capture_rng_unchanged':True,'group_metric_checks':True,'native_preprocessor_mutation_isolated':True}
    (OUT/'cpu-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
