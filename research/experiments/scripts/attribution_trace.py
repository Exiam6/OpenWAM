"""Passive planner/physical-state recorder for fixed-command two-step probes."""
import inspect
import numpy as np
from repeat_common import array_hash,payload_hash
from trajectory_trace import TrajectoryTrace

def pack(value):
    if isinstance(value,np.ndarray):
        a=np.array(value,copy=True);assert np.isfinite(a).all()
        return {'dtype':str(a.dtype),'shape':list(a.shape),'values':a.tolist(),'sha256':array_hash(a)}
    if isinstance(value,np.generic):return pack(value.item())
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,dict):return {str(k):pack(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [pack(v)for v in value]
    if hasattr(value,'p')and hasattr(value,'q'):return {'p':pack(np.asarray(value.p)),'q':pack(np.asarray(value.q))}
    raise TypeError(f'unrecorded value type: {type(value)}')

def snapshot(env):
    robot=env.robot
    actual={};ee={}
    for side in ['left','right']:
        entity=getattr(robot,side+'_entity')
        actual[side]={'qpos':pack(entity.get_qpos()),'qvel':pack(entity.get_qvel()),'root_pose':pack(entity.get_root_pose())}
        ee[side]=pack(getattr(robot,side+'_ee').global_pose)
    objects={name:pack(getattr(env,name).get_pose())for name in ['bottle1','bottle2']}
    targets=pack(np.asarray(robot.get_left_arm_jointState()+robot.get_right_arm_jointState()))
    result={'actual_articulations':actual,'ee_link_poses':ee,'object_poses':objects,'joint_drive_targets':targets,'communication_flag':bool(robot.communication_flag)}
    result['sha256']=payload_hash(result)
    return result

class AttributionTrace(TrajectoryTrace):
    def take_action(self,original,action,*args,**kwargs):
        env=self.env;row=self.current;assert row['step']<2 and not env.robot.communication_flag
        row['before_execution']=snapshot(env);row['planner_calls']=[]
        planners=[getattr(env.robot,side+'_planner')for side in ['left','right']]
        assert planners[0]is not planners[1]
        restorations=[]
        try:
            for side,planner in zip(['left','right'],planners):
                old=planner.plan_path;restorations.append((planner,old))
                def wrapped(*a,_old=old,_side=side,**kw):
                    bound=inspect.signature(_old).bind(*a,**kw);bound.apply_defaults()
                    inputs=pack(bound.arguments);result=_old(*a,**kw)
                    row['planner_calls'].append({'side':_side,'inputs':inputs,'inputs_sha256':payload_hash(inputs),'result':pack(result),'result_sha256':payload_hash(pack(result))})
                    return result
                planner.plan_path=wrapped
            result=super().take_action(original,action,*args,**kwargs)
            row['after_execution']=snapshot(env)
            assert [x['side']for x in row['planner_calls']]==['left','right']
            return result
        finally:
            for planner,old in restorations:planner.plan_path=old

class FixedCommandTransport:
    def __init__(self,trace,commands):self.trace=trace;self.commands=commands
    def reset(self):return {'type':'reset_ack'}
    def predict(self,payload):
        row=self.trace.current;assert row is not None and row['step']in [0,1]
        action=self.commands[str(row['seed'])][row['step']]['server_action']
        return {'action':action,'step':row['step']+1}
