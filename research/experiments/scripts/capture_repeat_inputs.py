"""Capture zero-action initial requests using the exact native payload builder."""
import base64,io,json,os,random,sys
from pathlib import Path
import numpy as np
from PIL import Image
from norm_client import install_replay
from robustness_client import bootstrap_robotwin_module,_install_test_num_override
from robustness_common import digest,noise_image
from repeat_common import payload_hash
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/repeatability-20260921'

class RecordingTransport:
    def __init__(self):self.payloads=[]
    def reset(self):return {'type':'reset_ack'}
    def predict(self,payload):
        self.payloads.append(payload)
        # Discarded by capture; no environment action is ever executed.
        return {'action':[0.]*20}

def make_recording_model(interface):
    m=interface.ModelClient.__new__(interface.ModelClient)
    m._send_state=True;m._state_dim=20;m._action_indices=None;m._action_type='ee'
    m._task_description='';m._debug=False;m._episode=-1;m._step=0
    m._client=RecordingTransport()
    return m

def capture_payload(interface,model,observation,instruction):
    cams=observation['observation'];before=len(model._client.payloads)
    model.step({'cams':{'head':cams['head_camera']['rgb'],'left':cams.get('left_camera',{}).get('rgb'),'right':cams.get('right_camera',{}).get('rgb')},'lang':instruction,'state':interface._extract_proprio(model,observation)})
    assert len(model._client.payloads)==before+1
    return model._client.payloads[-1]

def main():
    scenes=json.loads((OUT/'scenes.json').read_text())['scenes'];by_seed={r['seed']:r for r in scenes}
    module=install_replay(bootstrap_robotwin_module(),scenes);_install_test_num_override(module)
    import openwam2robotwin_interface as interface
    active={};records=[];old_class=module.class_decorator
    def create(task):
        env=old_class(task);setup=env.setup_demo
        def reset(*args,**kw):
            active['seed']=int(kw['seed']);kw['eval_video_log']=False
            return setup(*args,**kw)
        env.setup_demo=reset
        return env
    def capture(env,model,observation):
        seed=active['seed'];row=by_seed[seed];assert env.take_action_cnt==0
        assert env.get_instruction()==row['instruction']
        assert digest(observation['observation']['head_camera']['rgb'])==row['initial_head_sha256']
        assert digest(interface._extract_proprio(model,observation))==row['initial_state_sha256']
        clean=observation['observation'];np_state=np.random.get_state();py_state=random.getstate()
        for condition in ('clean','noise'):
            head=clean['head_camera']['rgb'] if condition=='clean' else noise_image(clean['head_camera']['rgb'],seed,0,sigma=.20)
            altered={**observation,'observation':{**clean,'head_camera':{**clean['head_camera'],'rgb':head}}}
            payload=capture_payload(interface,model,altered,row['instruction'])
            decoded={k:np.array(Image.open(io.BytesIO(base64.b64decode(v))))for k,v in payload['images'].items() if v is not None}
            assert np.array_equal(decoded['head_camera'],head)
            for src,dst in [('left_camera','left_wrist_camera'),('right_camera','right_wrist_camera')]:assert np.array_equal(decoded[dst],clean[src]['rgb'])
            name=f'{condition}-{seed}';path=OUT/'inputs'/f'{name}.json';assert not path.exists();path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(payload,allow_nan=False)+'\n')
            records.append({'name':name,'seed':seed,'condition':condition,'payload_sha256':payload_hash(payload),'raw_head_sha256':digest(clean['head_camera']['rgb']),'state_sha256':digest(interface._extract_proprio(model,observation)),'policy_head_sha256':digest(head),'actions_executed':0})
        assert all(np.array_equal(a,b)for a,b in zip(np_state,np.random.get_state())) and py_state==random.getstate()
        # End this capture-only inner loop without advancing physics or executing a dummy action.
        env.step_lim=0
        print('CAPTURED_ZERO_ACTION_INPUTS',seed,flush=True)
    original_decorator=module.eval_function_decorator
    def decorator(policy,name):
        if name=='get_model':return lambda args:make_recording_model(interface)
        if name=='eval':return capture
        return original_decorator(policy,name)
    module.class_decorator=create;module.eval_function_decorator=decorator
    module.main(module.parse_args_and_config())
    assert len(records)==6 and {r['seed']for r in records}==set(by_seed)
    (OUT/'capture-summary.json').write_text(json.dumps({'role':'zero-action input capture, not policy success evaluation','all_initial_hashes_match':True,'all_png_roundtrips_exact':True,'ambient_rng_unchanged':True,'records':records},indent=2)+'\n')
if __name__=='__main__':main()
