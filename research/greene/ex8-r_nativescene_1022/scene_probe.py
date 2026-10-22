"""One original native setup/camera sample. No policy or expert rollout or score."""
import importlib.util,json,math,os,sys,hashlib
from pathlib import Path
from native_adapter import verify_device
class SceneCaptured(BaseException):pass

def capture(obs,extract,write):
 state=extract(obs)
 assert tuple(state.shape)==(20,) and all(math.isfinite(float(x)) for x in state),'Invalid native20D state'
 images={}
 for name in ['head_camera','left_camera','right_camera']:
  rgb=obs['observation'][name]['rgb']
  assert len(rgb.shape)==3 and rgb.shape[2]==3 and all(v>0 for v in rgb.shape),'Invalid RGB shape'
  assert str(rgb.dtype)=='uint8' and float(rgb.std())>0,'Invalid or constant RGB'
  images[name]=dict(shape=list(rgb.shape),dtype=str(rgb.dtype),std=float(rgb.std()),path=str(write(name,rgb)))
 return dict(state=state.tolist(),images=images)

def drive(wrapper,argv,expected,output,capture_observation):
 original_bootstrap=wrapper.bootstrap_robotwin_module
 original_prewarm=wrapper._prewarm_cuda_for_curobo
 restores=[];proof={}
 from prewarm import checked_prewarm
 wrapper._prewarm_cuda_for_curobo=lambda:checked_prewarm(original_prewarm)
 def bootstrap():
  module=original_bootstrap()
  original_function=module.eval_function_decorator;original_eval=module.eval_policy
  def function(policy,name):
   if name=='get_model':return lambda _:None
   return original_function(policy,name)
  def inspect_scene(task,env,args,model,seed,**kwargs):
   assert task=='adjust_bottle' and args['task_config']=='demo_clean' and seed==100000
   assert model is None,'Policy model forbidden in scene-only probe'
   try:
    native_args=dict(args,eval_mode=True)
    env.setup_demo(now_ep_num=0,seed=seed,is_test=True,**native_args)
    device=verify_device(env.scene.render_system.device,expected)
    proof.update(task=task,mode='demo_clean',candidate_seed=seed,device=device,
                 observation=capture_observation(env.get_obs()),policy_loaded=False,expert_rollout=False,scored_episodes=0)
   finally:env.close_env()
   raise SceneCaptured()
  module.eval_function_decorator=function;module.eval_policy=inspect_scene
  restores.extend([(module,'eval_function_decorator',original_function),(module,'eval_policy',original_eval)])
  return module
 wrapper.bootstrap_robotwin_module=bootstrap
 try:
  try:wrapper.main(argv)
  except SceneCaptured:pass
  else:raise RuntimeError('Native scene diagnostic callback not reached')
  assert proof and proof['scored_episodes']==0
  return proof
 finally:
  for obj,name,value in reversed(restores):setattr(obj,name,value)
  wrapper.bootstrap_robotwin_module=original_bootstrap;wrapper._prewarm_cuda_for_curobo=original_prewarm

def run_scene(expected,output):
 root=Path('/home/zifanz4/openwam-runtime/ex8-a_officialrepro_1021')
 path=root/'official/benchmarks/robotwin/eval_policy_wrapper.py'
 spec=importlib.util.spec_from_file_location('scene_native_wrapper',path);wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
 output=Path(output);images=output/'native-images';images.mkdir(exist_ok=False)
 from PIL import Image
 def write(name,rgb):
  target=images/(name+'.png');Image.fromarray(rgb).save(target);return target
 def extract(obs):
  from openwam2robotwin_interface import _extract_eef_proprio
  return _extract_eef_proprio(obs)
 argv=['--config',str(root/'official/benchmarks/robotwin/policy_config.yml'),'--overrides','--task_name','adjust_bottle','--task_config','demo_clean','--ckpt_setting','ex8r-scene-only','--seed','0','--policy_name','openwam2robotwin_interface']
 updates=dict(ROBOTWIN_PATH=str(root/'RoboTwin'),ROBOTWIN_RUNTIME_ROOT=str(output/'simulator'))
 old={k:os.environ.get(k) for k in updates};cwd=Path.cwd();oldargv=list(sys.argv);os.environ.update(updates)
 # Native module resolves the same public policy module before setup; no socket is opened.
 sys.path.insert(0,str(root/'official/benchmarks/robotwin'))
 try:
  proof=drive(wrapper,argv,expected,output,lambda obs:capture(obs,extract,write))
  proof['image_sha256']={v['path']:hashlib.sha256(Path(v['path']).read_bytes()).hexdigest() for v in proof['observation']['images'].values()}
  return proof
 finally:
  os.chdir(cwd);sys.argv=oldargv
  for k,v in old.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
