"""cross-cluster-rootcause-20261003 §1: Torch environment inventory (bench env; one tiny rt+oidn render to record loaded libs)."""
import json,os,subprocess,sys
def sh(c):
 try:return subprocess.run(c,shell=True,capture_output=True,text=True,timeout=120).stdout.strip()
 except Exception as e:return f'ERR {e}'
out={'host':sh('hostname'),'slurm_job':os.environ.get('SLURM_JOB_ID'),
 'nvidia_smi':sh('nvidia-smi --query-gpu=name,driver_version,compute_cap,pci.bus_id --format=csv,noheader'),
 'nvidia_smi_header':sh('nvidia-smi | head -4'),
 'nvcc':sh(os.environ.get('CUDA_HOME','')+'/bin/nvcc --version | tail -2'),
 'policy_env':sh('/scratch/zz4330/conda/envs/openwam-policy/bin/python -c "import torch;print(torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_capability())"'),
 'bench_python':sys.version,
 'env_vars':{k:v for k,v in os.environ.items() if k.startswith(('SAPIEN','VK_','ROBOTWIN','LD_LIBRARY','NVIDIA','CUDA','__GLX','OIDN'))},
 'vulkan_icd':sh('ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null; cat /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null'),
 'run_baseline_sets_planner_fallback':'ROBOTWIN_ENABLE_PLANNER_FALLBACK=\'1\' in research/greene/baseline-official/scripts/run_baseline.py (sha256 43becf07... == run-record of 18771822)'}
try:
 import torch;out['bench_torch']=[torch.__version__,torch.version.cuda]
except Exception as e:out['bench_torch']=f'ERR {e}'
import sapien,numpy as np
out['sapien']=sapien.__version__;od=os.path.join(os.path.dirname(sapien.__file__),'oidn_library');out['sapien_oidn_library']=sorted(os.listdir(od)) if os.path.isdir(od) else None
# Same render settings as RoboTwin envs/_base_task.py:214-217
sapien.render.set_camera_shader_dir('rt');sapien.render.set_ray_tracing_samples_per_pixel(32);sapien.render.set_ray_tracing_path_depth(8);sapien.render.set_ray_tracing_denoiser('oidn')
sc=sapien.Scene();sc.set_ambient_light([0.5,0.5,0.5]);sc.add_directional_light([0,1,-1],[0.5,0.5,0.5]);sc.add_ground(0)
b=sc.create_actor_builder();b.add_box_visual(half_size=[0.1,0.1,0.1],material=[0.8,0.2,0.2]);b.build_kinematic().set_pose(sapien.Pose([0,0,0.1]))
cam=sc.add_camera('c',320,240,1.0,0.01,10);cam.set_pose(sapien.Pose([-1,0,0.5]));sc.update_render();cam.take_picture();img=cam.get_picture('Color')
out['render_ok']=bool(np.isfinite(img).all());out['render_mean']=float(img[...,:3].mean())
out['render_device']=sh('true') or str(getattr(sapien.render,'get_device_summary',lambda:'n/a')())
maps=open('/proc/self/maps').read().split('\n');libs=sorted({l.split()[-1] for l in maps if any(s in l for s in ('OpenImageDenoise','vulkan','nvidia','libcuda','optix'))})
out['loaded_libs']=libs
print(json.dumps(out,indent=1))
