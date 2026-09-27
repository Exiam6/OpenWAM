"""Native eval-mode expert scene collection; no learned-policy outcomes."""
import datetime,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path('/data02/zifanz4/openwam-experiments')
RUN=ROOT/'endtoend-20260923/scenes'

def pci(s):
 a,b,c=s.split(':');d,e=c.split('.');return tuple(int(x,16) for x in [a,b,d,e])
def capture_initial(env,dest,phase):
 device=env.scene.render_system.device
 assert pci(device.pci_string)==pci(os.environ['EXPECTED_RENDER_PCI'])
 planners={side:type(getattr(env.robot,side+'_planner',None)).__name__ for side in ['left','right']}
 if phase=='plan':assert all(v=='CuroboPlanner' for v in planners.values()),planners
 obs=env.get_obs()
 from PIL import Image
 import openwam2robotwin_interface as interface
 states=np.asarray(interface._extract_eef_proprio(obs),dtype=np.float32)
 images={k:obs['observation'][k]['rgb'] for k in ['head_camera','left_camera','right_camera']}
 poses={}
 for k,v in vars(env).items():
  if k.startswith('_') or not callable(getattr(v,'get_pose',None)):continue
  try:
   pose=v.get_pose();poses[k]={'p':np.asarray(pose.p).tolist(),'q':np.asarray(pose.q).tolist()}
  except (AttributeError,TypeError):pass
 record={'state':states.tolist(),'poses':poses,'image_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in images.items()},'renderer_pci':device.pci_string,'planners':planners,'eval_mode':bool(env.eval_mode)}
 write(dest/f'initial-{phase}.json',record)
 if phase=='replay':
  plan=json.loads((dest/'initial-plan.json').read_text())
  assert np.allclose(plan['state'],states,rtol=0,atol=1e-6),'Expert plan/replay initial states differ'
  assert plan['poses'].keys()==poses.keys()
  for k in poses:
   for component in ['p','q']:assert np.allclose(plan['poses'][k][component],poses[k][component],rtol=0,atol=1e-6)
  for k,v in images.items():Image.fromarray(v).save(dest/f'initial-{k}.png')

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)

def worker(task,seed):
    dest=RUN/task/f'seed-{seed}';dest.mkdir(parents=True,exist_ok=True)
    result={'task':task,'seed':seed,'role':'expert_only_independent_confirmation','accepted':False,'started_at':datetime.datetime.now().astimezone().isoformat()}
    start=time.monotonic();env=None
    try:
        os.environ['ROBOTWIN_RUNTIME_ROOT']=str(dest/'runtime')
        sys.path.insert(0,str(ROOT/'endtoend-20260923/OpenWAM/benchmarks/robotwin'))
        from eval_policy_wrapper import bootstrap_robotwin_module
        mod=bootstrap_robotwin_module();import yaml,h5py
        args=yaml.safe_load(Path('task_config/demo_clean.yml').read_text())
        emb=yaml.safe_load(Path('task_config/_embodiment_config.yml').read_text())['aloha-agilex']['file_path']
        ec=mod.get_embodiment_config(emb)
        camera=mod.get_camera_config(args['camera']['head_camera_type'])
        args.update(task_name=task,task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=emb,right_robot_file=emb,left_embodiment_config=ec,right_embodiment_config=ec,dual_arm_embodied=True,head_camera_h=camera['h'],head_camera_w=camera['w'],save_path=str(dest),need_plan=True,save_data=False,render_freq=0,eval_mode=True,is_test=True)
        write(dest/'collection-config.json',args)
        env=mod.class_decorator(task);env.setup_demo(now_ep_num=0,seed=seed,**args);capture_initial(env,dest,'plan' if args['need_plan'] else 'replay')
        info=env.play_once();ok=bool(env.plan_success and env.check_success());result['expert_plan_success']=ok
        if not ok:
            result['outcome']='expert_infeasible';return
        import random
        random_state=random.getstate();numpy_state=np.random.get_state()
        try:
            random.seed(seed);np.random.seed(seed)
            descriptions=mod.generate_episode_descriptions(task,[info['info']],5)
            result['instruction']=descriptions[0]['unseen'][0]
        finally:random.setstate(random_state);np.random.set_state(numpy_state)
        env.save_traj_data(0);env.close_env();env=None
        args.update(need_plan=False,save_data=True)
        env=mod.class_decorator(task);env.setup_demo(now_ep_num=0,seed=seed,**args);capture_initial(env,dest,'plan' if args['need_plan'] else 'replay')
        traj=env.load_tran_data(0);args.update(left_joint_path=traj['left_joint_path'],right_joint_path=traj['right_joint_path']);env.set_path_lst(args)
        info=env.play_once();ok=bool(env.check_success());result['expert_replay_success']=ok
        env.close_env();env.merge_pkl_to_hdf5_video()
        path=dest/'data/episode0.hdf5'
        with h5py.File(path,'r') as f:
            n=len(f['endpose/left_endpose']);assert n>=33
            for key in ['observation/head_camera/rgb','observation/left_camera/rgb','observation/right_camera/rgb','endpose/right_endpose','endpose/left_gripper','endpose/right_gripper']:
                assert len(f[key])==n,key
        env.remove_data_cache();env=None
        result.update(accepted=ok,outcome='accepted' if ok else 'expert_replay_failed',frames=n,hdf5=str(path),hdf5_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scene_info=info)
    except Exception as exc:
        from envs.utils.create_actor import UnStableError
        result.update(outcome='unstable_scene' if isinstance(exc,UnStableError) else 'infrastructure_failure',error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        if env is not None:
            try:env.close_env()
            except Exception:pass
        result['elapsed_seconds']=time.monotonic()-start;write(dest/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]))
