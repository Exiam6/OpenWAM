"""Fixed-seed native expert collection; never calls a learned policy."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
TASKS=['adjust_bottle','handover_block','place_object_basket']
ROOT=Path('/data02/zifanz4/openwam-experiments')
RUN=ROOT/'results/layers-20260922/fresh'
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def worker(task,seed):
    dest=RUN/task/f'seed-{seed}';dest.mkdir(parents=True,exist_ok=True)
    result={'task':task,'seed':seed,'role':'expert_only_independent_confirmation','accepted':False,'started_at':datetime.datetime.now().astimezone().isoformat()}
    start=time.monotonic();env=None
    try:
        os.environ['ROBOTWIN_RUNTIME_ROOT']=str(dest/'runtime')
        sys.path.insert(0,str(ROOT/'OpenWAM/benchmarks/robotwin'))
        from eval_policy_wrapper import bootstrap_robotwin_module
        mod=bootstrap_robotwin_module();import yaml,h5py
        args=yaml.safe_load(Path('task_config/demo_clean.yml').read_text())
        emb=yaml.safe_load(Path('task_config/_embodiment_config.yml').read_text())['aloha-agilex']['file_path']
        ec=mod.get_embodiment_config(emb)
        camera=mod.get_camera_config(args['camera']['head_camera_type'])
        args.update(task_name=task,task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=emb,right_robot_file=emb,left_embodiment_config=ec,right_embodiment_config=ec,dual_arm_embodied=True,head_camera_h=camera['h'],head_camera_w=camera['w'],save_path=str(dest),need_plan=True,save_data=False,render_freq=0)
        write(dest/'collection-config.json',args)
        env=mod.class_decorator(task);env.setup_demo(now_ep_num=0,seed=seed,**args)
        info=env.play_once();ok=bool(env.plan_success and env.check_success());result['expert_plan_success']=ok
        if not ok:
            result['outcome']='expert_infeasible';return
        env.save_traj_data(0);env.close_env();env=None
        args.update(need_plan=False,save_data=True)
        env=mod.class_decorator(task);env.setup_demo(now_ep_num=0,seed=seed,**args)
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

def parent():
    RUN.mkdir(parents=True,exist_ok=True);assert not (RUN/'manifest-v2.json').exists(),'Do not regenerate completed fresh dataset'
    started=time.monotonic();attempts=[];accepted={task:[] for task in TASKS}
    deadline=datetime.datetime.fromisoformat('2026-09-22T15:25:23.153134-05:00')
    for ti,task in enumerate(TASKS):
        for offset in range(40):
            if datetime.datetime.now().astimezone()>=deadline:raise TimeoutError('Original fresh-collection allocation reached; no extension')
            seed=600000+ti*1000+offset;dest=RUN/task/f'seed-{seed}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():
                res=json.loads((dest/'result.json').read_text());res['reused_without_execution']=True
                if res['outcome']=='infrastructure_failure' and res.get('error','').startswith('UnStableError('):
                    res['original_outcome']=res['outcome'];res['outcome']='unstable_scene';res['classification_amendment']='v2: native scene feasibility failure, original result file unchanged'
                assert res['outcome']!='infrastructure_failure',res
                attempts.append(res)
                if res.get('accepted'):accepted[task].append(res)
                continue
            with (dest/'worker.log').open('w') as f:
                try:r=subprocess.run([sys.executable,__file__,'--task',task,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,timeout=min(600,max(1,(deadline-datetime.datetime.now().astimezone()).total_seconds())));code=r.returncode
                except subprocess.TimeoutExpired:code=124
            p=dest/'result.json';res=json.loads(p.read_text()) if p.exists() else {'task':task,'seed':seed,'outcome':'infrastructure_failure','accepted':False}
            res['exit_code']=code;attempts.append(res)
            if res.get('accepted'):accepted[task].append(res)
            write(RUN/'progress-v2.json',{'attempts':attempts,'accepted_counts':{t:len(x) for t,x in accepted.items()},'elapsed_seconds':time.monotonic()-started})
            print(task,seed,res['outcome'],{t:len(x) for t,x in accepted.items()},flush=True)
            if code or res['outcome']=='infrastructure_failure':raise RuntimeError(f'Preserved infrastructure failure {task} {seed}; no retry')
            if len(accepted[task])==20:break
    write(RUN/'manifest-v2.json',{'protocol':'first20 expert-feasible complete episodes within40 consecutive seeds per task','attempts':attempts,'accepted':accepted,'complete':all(len(x)==20 for x in accepted.values()),'no_policy_scoring':True,'elapsed_seconds':time.monotonic()-started})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.task:worker(a.task,a.seed)
    else:parent()
