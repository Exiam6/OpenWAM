"""Finite paired-scene cohort: fixed protocol, native expert filter once, two GPU slots."""
import datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/data02/zifanz4/openwam-experiments');E=R/'endtoend-20260923';S=Path('/home/zifanz4/openwam-experiments/studies/endtoend-20260923');P=Path('/home/zifanz4/openwam-experiments/scripts/endtoend')
sys.path.insert(0,'/home/zifanz4/openwam-experiments/scripts/goalaux')
from null_grasp_classification_v2 import annotate
plan=json.loads((E/'protocol.json').read_text());deadline=datetime.datetime.fromisoformat(plan['fresh_scenes']['collection_deadline'])
def now():return datetime.datetime.now().astimezone()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def guard():
 for p in [S/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),p
 assert now()<deadline,'Prespecified collection deadline reached'
def preflight():
 import torch
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 torch.zeros(1,device='cuda')
 sys.path.insert(0,str(E/'OpenWAM/benchmarks/robotwin'))
 from eval_policy_wrapper import bootstrap_robotwin_module
 bootstrap_robotwin_module()
 import sapien,numpy as np
 from collect_scene import pci
 engine=sapien.Engine();renderer=sapien.SapienRenderer();engine.set_renderer(renderer)
 sapien.render.set_camera_shader_dir('rt');sapien.render.set_ray_tracing_samples_per_pixel(32);sapien.render.set_ray_tracing_path_depth(8);sapien.render.set_ray_tracing_denoiser('oidn')
 scene=engine.create_scene();dev=scene.render_system.device;assert pci(dev.pci_string)==pci(os.environ['EXPECTED_RENDER_PCI'])
 scene.set_ambient_light([.5,.5,.5]);scene.add_ground(0);cam=scene.add_camera('preflight',64,64,1,.1,10);cam.set_pose(sapien.Pose([0,0,1]));scene.step();scene.update_render();cam.take_picture();rgba=cam.get_picture('Color');assert rgba.shape==(64,64,4) and np.isfinite(rgba).all()
 write(Path(os.environ['COHORT_SLOT_DIR'])/'preflight.json',{'passed':True,'time':now().isoformat(),'renderer_pci':dev.pci_string,'renderer_name':dev.name,'no_experimental_seed_used':True})
def collect(task,env):
 guard();root=E/'scenes'/task;root.mkdir(parents=True,exist_ok=True);assert not (root/'started.json').exists(),'No automatic restart'
 write(root/'started.json',{'time':now().isoformat(),'deadline':deadline.isoformat(),'gpu_uuid':env['CUDA_VISIBLE_DEVICES']})
 accepted=[];attempts=[];begin=time.monotonic()
 for offset in range(plan['fresh_scenes']['max_candidates_per_task']):
  guard();seed=plan['fresh_scenes']['seed_starts'][task]+offset;dest=root/f'seed-{seed}';dest.mkdir(exist_ok=False)
  with (dest/'worker.log').open('x') as log:
   cmd=[str(R/'benchmark-env/bin/python'),'-u',str(P/'collect_scene.py'),task,str(seed)]
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(dest/'owned-worker.json',{'pid':proc.pid,'group':proc.pid,'command':cmd})
   try:code=proc.wait(timeout=min(600,(deadline-now()).total_seconds()))
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
    code=124
  raw=json.loads((dest/'result.json').read_text()) if (dest/'result.json').exists() else {'task':task,'seed':seed,'accepted':False,'outcome':'infrastructure_failure','error':'Missing worker result'}
  raw['exit_code']=code;record=annotate(raw);attempts.append(record)
  if record.get('classification_amendment'):write(dest/'classification-amendment.json',record)
  if record.get('accepted'):
   assert record['frames']>=33 and record.get('instruction'),'Invalid complete scene'
   accepted.append(record)
  write(root/'progress.json',{'time':now().isoformat(),'accepted_count':len(accepted),'attempted_count':len(attempts),'accepted':accepted,'attempts':attempts,'elapsed_seconds':time.monotonic()-begin})
  print(task,seed,record['outcome'],len(accepted),flush=True)
  assert code==0 and record['outcome']!='infrastructure_failure','Failure retained; no automatic retry/replacement'
  if len(accepted)==plan['fresh_scenes']['accepted_per_task']:break
 write(root/'manifest.json',{'task':task,'accepted':accepted,'attempts':attempts,'complete':len(accepted)==50,'finished_at':now().isoformat(),'no_policy_scoring':True})
 assert len(accepted)==50,'Incomplete cohort: not released for confirmation scoring'
def parent(index):
 assert index in [5,7] and os.uname().nodename=='cm001';guard();slot=E/'collection-slots'/str(index);slot.mkdir(parents=True,exist_ok=True);assert not (slot/'start.json').exists()
 for group in [1,2]:assert json.loads((S/f'seed-exclusion-all-group{group}.json').read_text())['passed']
 frozen=json.loads((S/'collection-code-freeze.json').read_text())
 for f,want in frozen['files'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==want,f
 lock=(R/f'gpu-cm001-{index}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 row=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip().split(',');u,pci,mem,util,ecc=[x.strip() for x in row];assert int(mem)<100 and int(util)==0 and int(ecc)==0,row
 assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 write(slot/'start.json',{'time':now().isoformat(),'gpu_query':row,'deadline':deadline.isoformat()})
 env=dict(os.environ,WAM_ROOT=str(R),CUDA_VISIBLE_DEVICES=u,EXPECTED_RENDER_PCI=pci,ROBOTWIN_PATH=str(R/'benchmarks/RoboTwin'),ROBOTWIN_RUNTIME_ROOT=str(slot/'preflight-runtime'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1',TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),COHORT_SLOT_DIR=str(slot),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 code=1
 try:
  with (slot/'preflight.log').open('x') as log:
   pre=subprocess.run(['timeout','--signal=TERM','--kill-after=20s','600',str(R/'benchmark-env/bin/python'),'-u',str(P/'collect_cohort.py'),'preflight'],env=env,stdout=log,stderr=subprocess.STDOUT)
  assert pre.returncode==0 and json.loads((slot/'preflight.json').read_text())['passed'],'Renderer preflight failed'
  for task in (['adjust_bottle','handover_block'] if index==5 else ['place_object_basket']):collect(task,env)
  code=0
 finally:write(slot/'exit.json',{'time':now().isoformat(),'returncode':code})
if __name__=='__main__':
 if sys.argv[1]=='preflight':preflight()
 else:parent(int(sys.argv[1]))
