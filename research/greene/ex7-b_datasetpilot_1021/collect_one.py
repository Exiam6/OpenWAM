"""One fresh expert trajectory; dataset feasibility only, never a policy outcome."""
import argparse, datetime, hashlib, importlib.util, json, os, signal, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'template'))
from migration_render import pin_renderer,configure_robot_paths
from evaluate_policy import bootstrap_robotwin_module
from collect_scene import pci

def main(task,seed,split,out):
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();row=dict(task=task,seed=seed,split=split,status='started',started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def save():
        row['seconds']=time.monotonic()-start
        tmp=out/'status.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(out/'status.json')
    save();env=None
    def timeout(*_):raise TimeoutError('Fixed 180 second expert budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    try:
        pin_renderer();mod=bootstrap_robotwin_module();configure_robot_paths()
        import openwam2robotwin_interface as interface
        from sapien.wrapper.scene import Scene
        spec=importlib.util.spec_from_file_location('multiview',ROOT/'multiview.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        original=Scene.step;ticks={}
        def step(scene):
            result=original(scene);ticks[id(scene)]=ticks.get(id(scene),0)+1;return result
        Scene.step=step
        args=json.loads((ROOT/'template/collection-template.json').read_text())
        args.update(task_name=task,save_path=str(out),need_plan=True,save_data=False,collect_data=False,
                    eval_mode=False,is_test=False,render_freq=0,eval_video_save_dir=None)
        env=mod.class_decorator(task);env.setup_demo(now_ep_num=0,seed=seed,**args)
        assert pci(env.scene.render_system.device.pci_string)==pci(os.environ['EXPECTED_RENDER_PCI'])
        dt=float(env.scene.get_timestep());first=ticks.get(id(env.scene),0);frames=[];states=[];times=[]
        def capture():
            assert len(frames)<3000,'Fixed trajectory memory bound'
            obs=env.get_obs();q=np.asarray(interface._extract_eef_proprio(obs),dtype=np.float32)
            assert q.shape==(20,) and np.isfinite(q).all()
            cameras={k:Image.fromarray(obs['observation'][k]['rgb']) for k in ['head_camera','left_camera','right_camera']}
            canvas=helper.assemble_multiview_layout(cameras,list(cameras),384,320)
            frames.append(np.asarray(canvas));states.append(q);times.append(ticks.get(id(env.scene),0)-first)
        env._take_picture=capture;capture();env.play_once();capture()
        assert env.plan_success and env.check_success(),'Expert failure retained; no replacement seed'
        assert len(frames)>=33 and np.all(np.diff(times)>=0) and times[-1]>times[0]
        np.savez_compressed(out/'trajectory.npz',images=np.stack(frames),proprio=np.stack(states),physics_ticks=np.asarray(times),timestep=dt)
        Image.fromarray(frames[0]).save(out/'first.png')
        row.update(status='complete',expert_success=True,frames=len(frames),timestep=dt,
                   episode_seconds=(times[-1]-times[0])*dt,sha256=hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest())
    except Exception as exc:row.update(status='failed',error=repr(exc))
    finally:
        signal.alarm(0)
        if env is not None:
            try:env.close_env(clear_cache=True)
            except Exception as exc:row.update(status='failed',cleanup_error=repr(exc))
        save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task');p.add_argument('--seed',type=int);p.add_argument('--split');p.add_argument('--out',type=Path);a=p.parse_args();main(a.task,a.seed,a.split,a.out)
