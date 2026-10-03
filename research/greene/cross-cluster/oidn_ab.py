"""Render-only OIDN probe on Torch (no policy, no scoring): setup_demo + get_obs on the 60 baseline scenes,
initial-frame MAE vs the reference PNGs exactly as evaluate_policy.py computes it. VARIANT (env):
 A = as deployed; B = ray-tracing denoiser forced to 'none' (simulates OIDN not running);
 C = run under LD_PRELOAD of OIDN 2.3.3 (set by the sbatch). Saves every actual render."""
import json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
SCRIPTS='/scratch/zz4330/OpenWAM/research/greene/baseline-official/scripts';sys.path.insert(0,SCRIPTS)
sys.path.insert(0,'/scratch/zz4330/OpenWAM/benchmarks/robotwin')
E=Path('/scratch/zz4330/openwam-runtime/assets-source/temporal-20260926');TASKS=['adjust_bottle','handover_block','place_object_basket']
variant=os.environ['VARIANT'];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
from migration_render import pin_renderer
pin_renderer()
import sapien
if variant=='B':
 _orig=sapien.render.set_ray_tracing_denoiser;sapien.render.set_ray_tracing_denoiser=lambda name:_orig('none')
from eval_policy_wrapper import bootstrap_robotwin_module
mod=bootstrap_robotwin_module()
rows=[];t0=time.monotonic()
for task in TASKS:
 manifest=json.loads((E/'scenes'/task/'manifest.json').read_text());env=mod.class_decorator(task)
 for i,scene in enumerate(manifest['accepted']):
  seed=int(scene['seed']);src=E/'scenes'/task/f'seed-{seed}';dest=out/task/f'seed-{seed}';dest.mkdir(parents=True,exist_ok=True)
  args=json.loads((src/'collection-config.json').read_text());args.update(save_path=str(dest),need_plan=True,save_data=False,eval_mode=True,is_test=True,render_freq=0,eval_video_save_dir=None)
  try:
   env.setup_demo(now_ep_num=0,seed=seed,**args);obs=env.get_obs();mae={}
   for key in ['head_camera','left_camera','right_camera']:
    actual=obs['observation'][key]['rgb'];ref=np.asarray(Image.open(src/f'initial-{key}.png'))
    mae[key]=float(np.abs(actual.astype(float)-ref.astype(float)).mean());Image.fromarray(actual).save(dest/f'{key}.png')
   row={'task':task,'seed':seed,'mae':mae,'max_mae':max(mae.values())}
  except Exception as exc:row={'task':task,'seed':seed,'error':repr(exc)}
  finally:env.close_env(clear_cache=((i+1)%5==0))
  rows.append(row);print(json.dumps(row),flush=True)
maps=open('/proc/self/maps').read().split('\n');libs=sorted({l.split()[-1] for l in maps if 'OpenImageDenoise' in l})
m=[r['max_mae'] for r in rows if 'max_mae' in r]
summary={'variant':variant,'job':os.environ.get('SLURM_JOB_ID'),'ld_preload':os.environ.get('LD_PRELOAD'),'oidn_libs_loaded':libs,'scenes':len(rows),'errors':sum('error' in r for r in rows),
 'max_mae':{'min':min(m),'median':float(np.median(m)),'max':max(m)} if m else None,'over_gate_1.0':sum(x>1. for x in m),'seconds':time.monotonic()-t0,'rows':rows}
(out/'summary.json').write_text(json.dumps(summary,indent=1));print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=1))
