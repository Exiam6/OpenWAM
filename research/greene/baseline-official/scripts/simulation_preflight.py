"""Training-scene-only end-to-end smoke, using the same helpers as heldout evaluation."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from evaluate_policy import bootstrap_robotwin_module,rotate_head_camera,gripper_contacts,noise_observation,write
from collect_scene import pci
from migration_render import pin_renderer
pin_renderer()
E=Path('/scratch/zz4330/openwam-runtime/assets-source/temporal-20260926')
p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
mod=bootstrap_robotwin_module();import openwam2robotwin_interface as interface
import yaml
interface._STEP_LIM_OVERRIDES={}
args=yaml.safe_load(Path('task_config/demo_clean.yml').read_text());emb=yaml.safe_load(Path('task_config/_embodiment_config.yml').read_text())['aloha-agilex']['file_path'];ec=mod.get_embodiment_config(emb);camera=mod.get_camera_config(args['camera']['head_camera_type'])
args.update(task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=emb,right_robot_file=emb,left_embodiment_config=ec,right_embodiment_config=ec,dual_arm_embodied=True,head_camera_h=camera['h'],head_camera_w=camera['w'],save_path=a.output,need_plan=True,save_data=False,render_freq=0,eval_mode=True,is_test=True,eval_video_save_dir=None)
model=interface.ModelClient(host='127.0.0.1',port=a.port,send_state=True,state_dim=20,request_timeout=120,action_type='ee')
cur={'step':0,'seed':1771,'action':None}
def send(payload):
 payload=dict(payload);payload.update(_study_step=cur['step'],_study_action_seed=cur['seed']);response=model._client.predict_once(payload);cur['action']=np.asarray(response['action']);return response
model._client.predict=send;results=[]
# Development-only seed10; no new heldout outcomes. No full old evaluation replay.
for task in ['adjust_bottle','handover_block','place_object_basket']:
 env=mod.class_decorator(task);args['task_name']=task
 try:
  args['need_plan']=True;env.setup_demo(now_ep_num=0,seed=10,**args);assert pci(env.scene.render_system.device.pci_string)==pci(os.environ['EXPECTED_RENDER_PCI']);assert env.eval_mode
  obs=env.get_obs();state=interface._extract_eef_proprio(obs);assert state.shape==(20,);limit=env.step_lim
  instruction={'adjust_bottle':'Lift the bottle.','handover_block':'Hand over the block.','place_object_basket':'Place the object in the basket.'}[task];env.set_instruction(instruction=instruction)
  example={'cams':{'head':obs['observation']['head_camera']['rgb'],'left':obs['observation']['left_camera']['rgb'],'right':obs['observation']['right_camera']['rgb']},'lang':instruction,'state':state}
  actions=[]
  for _ in range(2):
   interface.reset_model(model);cur['step']=0;action=model.step(example);assert action.size==20 and np.isfinite(action).all();actions.append(action)
  assert np.array_equal(*actions),'WS per-episode RNG/reset differs'
  multiprompt=None
  if task=='adjust_bottle':
   reference=actions[0].copy()
   for case in range(33):
    interface.reset_model(model);cur['step']=0
    probe=dict(example,lang='Lift the bottle. Training-only cache verification case '+str(case)+'.')
    got=model.step(probe);assert got.size==20 and np.isfinite(got).all()
   interface.reset_model(model);cur['step']=0;recomputed=model.step(example)
   assert np.array_equal(reference,recomputed),'WS action changed after33distinct prompts'
   multiprompt={'distinct_prompts':33,'all_finite':True,'post_eviction_action_exact':True}
   print('MULTIPROMPT_GATE_PASSED',flush=True)

  pose=rotate_head_camera(env,10);altered=env.get_obs();head_mae=float(np.abs(altered['observation']['head_camera']['rgb'].astype(float)-obs['observation']['head_camera']['rgb'].astype(float)).mean());assert head_mae>1.,'Head camera pose change did not alter actual rendering'
  assert np.array_equal(interface._extract_eef_proprio(altered),state),'Changing camera changed physical state'
  wrist_mae={k:float(np.abs(altered['observation'][k]['rgb'].astype(float)-obs['observation'][k]['rgb'].astype(float)).mean()) for k in ['left_camera','right_camera']};assert max(wrist_mae.values())<=1.,'Head perturbation changed wrist images'
  noisy=noise_observation(altered,.1,10,0);interface.reset_model(model);cur['step']=0
  interface.eval(env,model,noisy);assert env.take_action_cnt==1 and env.step_lim==limit and np.isfinite(cur['action']).all()
  rollout=None
  if task=='adjust_bottle':
   while env.take_action_cnt<limit and not env.eval_success:
    cur['step']=int(env.take_action_cnt)
    interface.eval(env,model,noise_observation(env.get_obs(),.1,10,cur['step']))
   rollout={'completed':True,'success':bool(env.eval_success),'steps':int(env.take_action_cnt),'scene':'development seed10 only'}
  contacts=gripper_contacts(env)
  for contact in env.scene.get_contacts():
   assert len(contact.bodies)==2
   for point in contact.points:assert np.isfinite(np.asarray(point.impulse)).all()
  # Same setup seed must reconstruct the same initial proprio/render before outcome scoring.
  env.close_env(clear_cache=True);env=mod.class_decorator(task);args['need_plan']=False;env.setup_demo(now_ep_num=0,seed=10,**args);again=env.get_obs();assert np.allclose(interface._extract_eef_proprio(again),state,rtol=0,atol=1e-6)
  repeat_mae={k:float(np.abs(again['observation'][k]['rgb'].astype(float)-obs['observation'][k]['rgb'].astype(float)).mean()) for k in ['head_camera','left_camera','right_camera']};assert max(repeat_mae.values())<=1.,repeat_mae
  results.append({'multi_prompt_gate':multiprompt,'task':task,'training_scene_seed':10,'native_step_limit':int(limit),'head_rotation':pose,'head_image_mae':head_mae,'wrist_mae':wrist_mae,'repeat_initial_mae':repeat_mae,'ws_reset_exact':True,'native_action_taken':True,'real_contact_api_checked':True,'contact_count':len(contacts),'full_development_rollout':rollout})
 finally:env.close_env(clear_cache=True)
model._client.close();write(Path(a.output)/'result.json',{'passed':True,'heldout_scenes_used':False,'profile_checkpoint_only':True,'tasks':results})
