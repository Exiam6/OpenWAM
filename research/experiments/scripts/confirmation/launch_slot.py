"""Own one idle GPU; hard limits and only this launcher's subprocess tree."""
import argparse, datetime, fcntl, json, os, subprocess, sys, traceback
from pathlib import Path
from worker import R, O, guards, write, now

def main(a):
    guards(); assert os.uname().nodename == 'cm009'
    deadline = datetime.datetime.fromisoformat(json.loads((O / 'window.json').read_text())['collection_deadline' if a.mode == 'collect' else 'deadline'])
    lock = (R / f'gpu-cm009-{a.index}.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    q = subprocess.check_output(['nvidia-smi','-i',str(a.index),'--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip()
    u, pci, mem, util, ecc = [x.strip() for x in q.split(',')]
    assert int(mem) < 100 and int(util) == 0 and int(ecc) == 0, q
    assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    dest = O / ('features' if a.mode == 'features' else ('preflight' if a.mode == 'preflight' else 'fresh')) / a.task; dest.mkdir(parents=True,exist_ok=True)
    if a.mode == 'features': dest = O / 'features'; dest.mkdir(exist_ok=True)
    (dest / 'launch-claim').mkdir()
    seconds = min(600 if a.mode == 'preflight' else 14400, int((deadline-now()).total_seconds()))
    # Parent enforces cutoff; extra30s only allows owned subprocess cleanup.
    if a.mode == "collect": seconds += 30
    assert seconds > 30
    write(dest / 'launch.json', {'started_at':now().isoformat(),'deadline':deadline.isoformat(),'gpu_query':q,'timeout_seconds':seconds,'cpu_threads':4})
    env = dict(os.environ, WAM_ROOT=str(R),CUDA_VISIBLE_DEVICES=u,EXPECTED_RENDER_PCI=pci,ROBOTWIN_PATH=str(R/'benchmarks/RoboTwin'),ROBOTWIN_RUNTIME_ROOT=str(dest/'preflight-runtime'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1',TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    command=[str(R/'benchmark-env/bin/python'),'-u',str(R/'scripts/confirmation/worker.py'),a.mode,'--task',a.task]
    if a.mode == 'features':
        env.update(LAYER_RUN=str(R/'results/layers-20260922'),PYTHONHOME=str(R/'assets/python310-runtime'),PYTHONPATH=str(R/'env/lib/python3.10/site-packages'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        command=[str(R/'env/bin/python'),'-u',str(R/'scripts/confirmation/features.py')]
    with (dest/'run.log').open('x') as log:
        run = subprocess.run(['timeout','--signal=TERM','--kill-after=20s',str(seconds)]+command,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(dest/'exit.json',{'finished_at':now().isoformat(),'exit_code':run.returncode})
    return run.returncode

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','collect','features']);p.add_argument('--task',required=True);p.add_argument('--index',required=True,type=int);a=p.parse_args()
    try: code=main(a)
    except Exception:
        dest=O/('features' if a.mode=='features' else ('preflight' if a.mode=='preflight' else 'fresh'))/a.task
        if a.mode == 'features': dest=O/'features'
        write(dest/'launch-failure.json',{'time':now().isoformat(),'traceback':traceback.format_exc()});code=1
    raise SystemExit(code)
