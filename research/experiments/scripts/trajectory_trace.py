"""Passive per-action recorder. Original arguments and return objects are retained."""
import gzip,json
from pathlib import Path
import numpy as np
from repeat_common import array_hash,payload_hash

CAMERAS=('head_camera','left_camera','right_camera')
def camera_hashes(observation):
    return {k:array_hash(observation['observation'][k]['rgb']) for k in CAMERAS}

class TrajectoryTrace:
    def __init__(self,out,initial_payload_hashes,extract_state):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.initial=initial_payload_hashes;self.extract_state=extract_state;self.current=None
    def begin(self,raw,altered,condition,seed,step):
        assert self.current is None
        state=np.asarray(self.extract_state(raw),dtype=np.float32)
        joint=np.asarray(raw['joint_action']['vector'])
        self.current={'seed':int(seed),'step':int(step),'condition':condition,'raw_cameras':camera_hashes(raw),'policy_cameras':camera_hashes(altered),'eef_state':state.tolist(),'eef_hash':array_hash(state),'joint_state':joint.tolist(),'joint_dtype':str(joint.dtype),'joint_hash':array_hash(joint),'requests':0,'actions':0}
    def predict(self,original,payload):
        row=self.current;assert row is not None and row['requests']==0
        sha=payload_hash(payload)
        if row['step']==0:assert sha==self.initial[(row['condition'],row['seed'])], 'initial complete request differs'
        body=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        path=self.out/'requests'/f"seed-{row['seed']}"/f"step-{row['step']:04d}.json.gz"
        path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
        path.write_bytes(gzip.compress(body,mtime=0))
        row.update({'request_sha256':sha,'request_file':str(path.relative_to(self.out)),'requests':1})
        response=original(payload)
        assert payload_hash(payload)==sha, 'client transport mutated canonical payload'
        assert int(response['step'])==row['step']+1, 'server action buffer counter mismatch'
        action=np.asarray(response['action'],dtype=np.float32);assert action.shape==(20,) and np.isfinite(action).all()
        row.update({'server_action':action.tolist(),'server_action_hash':array_hash(action),'server_step':int(response['step'])})
        return response
    def take_action(self,original,action,*args,**kwargs):
        row=self.current;assert row is not None and row['requests']==1 and row['actions']==0
        a=np.asarray(action);assert a.shape==(16,) and np.isfinite(a).all()
        row.update({'env_action':a.tolist(),'env_action_hash':array_hash(a),'env_action_dtype':str(a.dtype),'action_type':kwargs.get('action_type'),'actions':1})
        return original(action,*args,**kwargs)
    def finish(self,env):
        row=self.current;assert row is not None and row['requests']==row['actions']==1
        assert int(env.take_action_cnt)==row['step']+1
        row['after_action_count']=int(env.take_action_cnt);row['success_after_action']=bool(env.eval_success)
        with (self.out/f"seed-{row['seed']}-trace.jsonl").open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.current=None
    def abort(self,error):
        (self.out/'incomplete-trace.json').write_text(json.dumps({'error':str(error),'record':self.current},indent=2)+'\n')

def instrument_interface(interface,trace):
    old_get,old_eval=interface.get_model,interface.eval
    def get_model(args):
        model=old_get(args);predict=model._client.predict
        model._client.predict=lambda payload:trace.predict(predict,payload)
        return model
    def eval_step(env,model,observation):
        take=env.take_action
        env.take_action=lambda action,*a,**kw:trace.take_action(take,action,*a,**kw)
        try:
            result=old_eval(env,model,observation)
            trace.finish(env)
            return result
        except BaseException as error:
            trace.abort(error);raise
        finally:env.take_action=take
    interface.get_model=get_model;interface.eval=eval_step
