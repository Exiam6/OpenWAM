"""Copied native stack imports and empty-scene device gate; no task seeds."""
import datetime,hashlib,json,os,sys
from pathlib import Path
R=Path(os.environ['WAM_ROOT']);O=R/'results/goalaux-20260922/native-preflight'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
def pci(v):
 a,b,c=v.split(':');c,d=c.split('.');return tuple(int(t,16) for t in [a,b,c,d])
plan=json.loads((O/'protocol.json').read_text());assert not (O/'started.json').exists()
for f,h in plan['sha256'].items():assert sha(R/f)==h,f
write('started.json',{'time':datetime.datetime.now().astimezone().isoformat(),'protocol_sha256':sha(O/'protocol.json')})
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.cuda.get_device_name(0)=='NVIDIA L40S';torch.zeros(1,device='cuda')
sys.path.insert(0,str(R/'OpenWAM/benchmarks/robotwin'))
from eval_policy_wrapper import bootstrap_robotwin_module
mod=bootstrap_robotwin_module()
from curobo.wrap.reacher.motion_gen import MotionGen
import curobo,h5py,cv2,sapien,numpy as np,importlib.metadata
assert str(curobo.__file__).startswith(str(R/'goalaux-native/RoboTwin'))
engine=sapien.Engine();renderer=sapien.SapienRenderer();engine.set_renderer(renderer)
sapien.render.set_camera_shader_dir('rt');sapien.render.set_ray_tracing_samples_per_pixel(32);sapien.render.set_ray_tracing_path_depth(8);sapien.render.set_ray_tracing_denoiser('oidn')
scene=engine.create_scene();device=scene.render_system.device
assert pci(device.pci_string)==pci(os.environ['EXPECTED_RENDER_PCI']),(device.pci_string,os.environ['EXPECTED_RENDER_PCI'])
scene.set_ambient_light([.5,.5,.5]);scene.add_ground(0);camera=scene.add_camera('preflight',64,64,1,.1,10);camera.set_pose(sapien.Pose([0,0,1]));scene.step();scene.update_render();camera.take_picture();rgba=camera.get_picture('Color');assert rgba.shape==(64,64,4) and np.isfinite(rgba).all()
write('result.json',{'passed':True,'completed_at':datetime.datetime.now().astimezone().isoformat(),'renderer':{'name':device.name,'pci':device.pci_string,'cuda_id':device.cuda_id},'torch_device':torch.cuda.get_device_name(0),'versions':{k:importlib.metadata.version(k) for k in ['torch','sapien','h5py','numpy']},'camera_shape':list(rgba.shape),'native_bootstrap_imported':True,'curobo_imported':True,'no_task_scenes_or_confirmation_seeds_used':True,'planner_runtime_backend_gate':'still required for every scene'})
print('NATIVE_PREFLIGHT_PASS',flush=True)
