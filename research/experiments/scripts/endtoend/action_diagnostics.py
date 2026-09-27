"""Fixed expert-state one-step diagnostics; secondary, never checkpoint selection."""
import argparse,json,sys,time
from pathlib import Path
import cv2,h5py,numpy as np
from evaluate_policy import bootstrap_robotwin_module,noise_observation,write
E=Path('/data02/zifanz4/openwam-experiments/endtoend-20260923');S=Path('/home/zifanz4/openwam-experiments/studies/endtoend-20260923')

def rotation_matrix(six):
 # Native rot6d stores the first two matrix columns, flattened column-wise.
 x=np.asarray(six,dtype=float);a=x[:3];b=x[3:];a=a/max(np.linalg.norm(a),1e-12);b=b-np.dot(a,b)*a;b=b/max(np.linalg.norm(b),1e-12);return np.stack([a,b,np.cross(a,b)],axis=-1)
def errors(pred,target):
 p=np.asarray(pred);q=np.asarray(target);translation=float(np.mean((p[[0,1,2,10,11,12]]-q[[0,1,2,10,11,12]])**2));gripper=float(np.mean((p[[9,19]]-q[[9,19]])**2))
 angles=[]
 for sl in [slice(3,9),slice(13,19)]:
  rel=rotation_matrix(p[sl]).T@rotation_matrix(q[sl]);angles.append(float(np.degrees(np.arccos(np.clip((np.trace(rel)-1)/2,-1,1)))))
 return {'translation_mse_m2':translation,'rotation_mean_deg':float(np.mean(angles)),'gripper_mse_native_units':gripper}
def run(a):
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);assert not(out/'started.json').exists();write(out/'started.json',vars(a));assert json.loads((S/'fresh-cohort-integrity.json').read_text())['passed']
 bootstrap_robotwin_module();import openwam2robotwin_interface as interface
 from benchmarks.utils import action_conversion
 model=interface.ModelClient(host='127.0.0.1',port=a.port,send_state=True,state_dim=20,request_timeout=120,action_type='ee');current={'seed':0}
 def send(payload):
  payload=dict(payload);payload.update(_study_step=0,_study_action_seed=current['seed']);return model._client.predict_once(payload)
 model._client.predict=send;rows=[];begin=time.monotonic()
 try:
  for task in ['adjust_bottle','handover_block','place_object_basket']:
   manifest=json.loads((E/'scenes'/task/'manifest.json').read_text());assert manifest['complete']
   for scene in manifest['accepted']:
    assert not (S/'paused.json').exists() and not Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
    scene_seed=scene['seed']
    with h5py.File(scene['hdf5'],'r') as f:
     n=len(f['endpose/left_endpose']);indices=np.linspace(0,n-2,12,dtype=int)
     for frame in indices:
      def state(i):return action_conversion.robotwin_endpose_to_eef20d(f['endpose/left_endpose'][i],f['endpose/right_endpose'][i],f['endpose/left_gripper'][i],f['endpose/right_gripper'][i])
      s=state(frame);target=state(frame+1);obs={'observation':{k:{'rgb':cv2.imdecode(np.frombuffer(bytes(f[f'observation/{k}/rgb'][frame]),np.uint8),cv2.IMREAD_COLOR)} for k in ['head_camera','left_camera','right_camera']}}
      for sigma in [0.,.04,.10]:
       current['seed']=int(np.random.SeedSequence([scene_seed,a.seed,int(frame),829]).generate_state(1,dtype=np.uint32)[0]);altered=noise_observation(obs,sigma,scene_seed,int(frame));cams=altered['observation'];interface.reset_model(model)
       pred=model.step({'cams':{'head':cams['head_camera']['rgb'],'left':cams['left_camera']['rgb'],'right':cams['right_camera']['rgb']},'lang':scene['instruction'],'state':s});assert pred.size==20 and np.isfinite(pred).all()
       row={'task':task,'scene_seed':scene_seed,'training_seed':a.seed,'route':a.route,'frame':int(frame),'target_frame':int(frame+1),'sigma':sigma,'state':s.tolist(),'prediction':pred.tolist(),'target':target.tolist(),'prediction_errors':errors(pred,target),'hold_current_state_errors':errors(s,target),'gripper_closing_proxy':bool(np.any(target[[9,19]]<s[[9,19]]-1e-3)),'contact_annotation':'gripper-closing proxy; no physical contact claim for expert HDF5 frames'}
       with (out/'rows.jsonl').open('a') as dest:dest.write(json.dumps(row)+'\n')
       rows.append(row)
    write(out/'progress.json',{'predictions':len(rows),'last_task':task,'last_scene':scene_seed})
  assert len(rows)==5400;write(out/'summary.json',{'complete':True,'predictions':len(rows),'seconds':time.monotonic()-begin,'one_step_teacher_forced':True,'native_action_target':'raw state at current frame+1, matching native dataset offset','conditions':['clean','noise.04','noise.10'],'real_camera_yaw_not_available_in_saved_expert_images':True,'model_means':{k:float(np.mean([x['prediction_errors'][k] for x in rows])) for k in rows[0]['prediction_errors']}})
 finally:model._client.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--route',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);run(p.parse_args())
