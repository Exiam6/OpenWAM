"""Fresh paired RoboTwin rollouts, retaining every attempt and native success rule."""
import argparse,copy,datetime,hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
E=Path('/home/zifanz4/openwam-runtime/assets-source/temporal-20260926');S=Path('/home/zifanz4/openwam-runtime/temporal-new-20260928/resume-v1')
sys.path.insert(0,str(E/'OpenWAM/benchmarks/robotwin'))
from eval_policy_wrapper import bootstrap_robotwin_module

def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def noise_observation(obs,sigma,seed,step):
 if not sigma:return obs
 result=dict(obs);result['observation']=dict(obs['observation'])
 for i,key in enumerate(['head_camera','left_camera','right_camera']):
  camera=dict(obs['observation'][key]);rgb=np.asarray(camera['rgb'])
  rng=np.random.default_rng(np.random.SeedSequence([seed,step,i,701]))
  camera['rgb']=np.rint(np.clip(rgb.astype(np.float32)+rng.normal(0,255*sigma,rgb.shape),0,255)).astype(np.uint8)
  result['observation'][key]=camera
 return result

def rotate_head_camera(env,seed):
 import sapien,transforms3d
 cameras=env.cameras.static_camera_list
 heads=[c for c,n in zip(cameras,env.cameras.static_camera_name) if n=='head_camera'];assert len(heads)==1
 camera=heads[0];pose=camera.entity.get_pose();angle=(1 if seed%2==0 else -1)*np.pi/36
 q=transforms3d.quaternions.qmult(transforms3d.quaternions.axangle2quat([0,0,1],angle),pose.q)
 record={'before_p':pose.p.tolist(),'before_q':pose.q.tolist(),'after_p':pose.p.tolist(),'after_q':q.tolist(),'world_z_degrees':float(angle*180/np.pi)}
 camera.entity.set_pose(sapien.Pose(pose.p,q));env.scene.update_render()
 return record

def gripper_contacts(env):
 result=[];grippers=set(env.robot.gripper_name)
 for contact in env.scene.get_contacts():
  names=[b.entity.name for b in contact.bodies]
  if grippers.intersection(names):result.append({'bodies':names,'points':len(contact.points),'impulse_norm_sum':float(sum(np.linalg.norm(p.impulse) for p in contact.points))})
 return result

def run(a):
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);assert not (out/'started.json').exists()
 protocol=json.loads((E/'protocol.json').read_text());manifest=json.loads((E/'scenes'/a.task/'manifest.json').read_text())
 assert manifest['complete'] and len(manifest['accepted'])==20
 assert a.condition in protocol['evaluation']['conditions'] and a.seed in [42,43,44]
 # A training-only simulator smoke test must pass before any heldout policy outcome.
 gate=json.loads((S/f'simulation-preflight-{a.route}-{a.seed}.json').read_text());assert gate['passed']
 for file,want in gate['code_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==want,file
 assert json.loads((S/'fresh-cohort-integrity.json').read_text())['passed']
 from migration_render import pin_renderer,configure_robot_paths
 pin_renderer()
 mod=bootstrap_robotwin_module();configure_robot_paths();import openwam2robotwin_interface as interface
 from collect_scene import pci
 import sapien,transforms3d
 interface._STEP_LIM_OVERRIDES={}
 model=interface.ModelClient(host='127.0.0.1',port=a.port,send_state=True,state_dim=20,request_timeout=120,action_type='ee')
 current={'step':0,'action_seed':0,'last_action':None,'latency_ms':None}
 def send(payload):
  payload=dict(payload);payload.update(_study_step=current['step'],_study_action_seed=current['action_seed'])
  response=model._client.predict_once(payload);current['last_action']=response['action'];current['latency_ms']=response.get('latency_ms');return response
 model._client.predict=send
 env=mod.class_decorator(a.task);records=[];begin=time.monotonic()
 write(out/'started.json',{'time':datetime.datetime.now().astimezone().isoformat(),'task':a.task,'route':a.route,'policy_seed':a.seed,'condition':a.condition,'manifest_sha256':hashlib.sha256((E/'scenes'/a.task/'manifest.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((E/'protocol.json').read_bytes()).hexdigest()})
 def timed_out(*_):raise TimeoutError('Prespecified600second episode wall limit reached')
 signal.signal(signal.SIGALRM,timed_out)
 try:
  selected=set(json.loads(a.scene_seeds));assert selected and selected.issubset({int(x['seed']) for x in manifest['accepted']})
  for index,scene in enumerate(manifest['accepted']):
   if int(scene['seed']) not in selected:continue
   for pause in [S/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not pause.exists(),pause
   seed=int(scene['seed']);dest=out/f'seed-{seed}';dest.mkdir(exist_ok=False);t0=time.monotonic();row={'scene_seed':seed,'training_seed':a.seed,'route':a.route,'task':a.task,'condition':a.condition,'status':'started'}
   signal.alarm(600)
   try:
    src=E/'scenes'/a.task/f'seed-{seed}';args=json.loads((src/'collection-config.json').read_text());args.update(save_path=str(dest),need_plan=True,save_data=False,eval_mode=True,is_test=True,render_freq=0,eval_video_save_dir=None)
    env.setup_demo(now_ep_num=0,seed=seed,**args)
    assert pci(env.scene.render_system.device.pci_string)==pci(os.environ['EXPECTED_RENDER_PCI'])
    assert bool(env.eval_mode)
    obs=env.get_obs();ref=json.loads((src/'initial-replay.json').read_text());state=interface._extract_eef_proprio(obs)
    assert np.allclose(state,ref['state'],rtol=0,atol=1e-6),'Initial proprio mismatch'
    for key,pose in ref['poses'].items():
     got=getattr(env,key).get_pose()
     assert np.allclose(got.p,pose['p'],rtol=0,atol=1e-6) and np.allclose(got.q,pose['q'],rtol=0,atol=1e-6),'Initial scene pose mismatch'
    errors={}
    for key in ['head_camera','left_camera','right_camera']:
     actual=obs['observation'][key]['rgb'];reference=np.asarray(Image.open(src/f'initial-{key}.png'))
     errors[key]=float(np.abs(actual.astype(float)-reference.astype(float)).mean())
     row['initial_render_mae']=dict(errors)
     if errors[key]>1.:
      Image.fromarray(actual).save(dest/f'initial-actual-{key}.png')
      assert False,'Initial render MAE exceeds1uint8level'
    row.update(initial_render_mae=errors,initial_state=digest(state),step_limit=int(env.step_lim),instruction=scene['instruction'])
    if a.condition=='head_camera_yaw_5deg':
     row['camera_pose']=rotate_head_camera(env,seed)
    sigma={'gaussian_sigma_0.04':.04,'gaussian_sigma_0.10':.10}.get(a.condition,0.)
    env.set_instruction(instruction=scene['instruction']);interface.reset_model(model)
    current['action_seed']=int(np.random.SeedSequence([seed,a.seed,811]).generate_state(1,dtype=np.uint32)[0])
    with (dest/'trace.jsonl').open('x') as trace:
     while env.take_action_cnt<row['step_limit']:
      current['step']=int(env.take_action_cnt);obs=env.get_obs();before=interface._extract_eef_proprio(obs)
      altered=noise_observation(obs,sigma,seed,current['step']);interface.eval(env,model,altered)
      assert env.step_lim==row['step_limit'],'Native task limit changed'
      action=np.asarray(current['last_action']);assert action.size==20 and np.isfinite(action).all()
      contacts=gripper_contacts(env)
      trace.write(json.dumps({'step':current['step'],'proprio':np.asarray(before).tolist(),'action':action.tolist(),'success':bool(env.eval_success),'latency_ms':current['latency_ms'],'gripper_contacts':contacts})+'\n')
      if env.eval_success:break
    row.update(status='complete',success=bool(env.eval_success),steps=int(env.take_action_cnt),seconds=time.monotonic()-t0,action_seed=current['action_seed'])
   except Exception as exc:
    row.update(status='infrastructure_failure',error=repr(exc),seconds=time.monotonic()-t0);write(dest/'result.json',row)
    if str(exc)=='Initial render MAE exceeds1uint8level':
     records.append(row);print(json.dumps(row),flush=True);continue
    raise
   finally:
    signal.alarm(0);env.close_env(clear_cache=((index+1)%5==0))
   write(dest/'result.json',row);records.append(row);write(out/'progress.json',{'completed':len(records),'successes':sum(bool(x.get('success',False)) for x in records),'seconds':time.monotonic()-begin})
   print(json.dumps(row),flush=True)
  write(out/'summary.json',{'complete':len(records)==len(selected) and all(x['status']=='complete' for x in records),'all_selected_attempted':len(records)==len(selected),'technical_failures':sum(x['status']!='complete' for x in records),'records':records,'successes':sum(bool(x.get('success',False)) for x in records),'episodes':len(records),'seconds':time.monotonic()-begin})
 finally:model._client.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene-seeds',required=True);p.add_argument('--task',required=True);p.add_argument('--route',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);run(p.parse_args())
