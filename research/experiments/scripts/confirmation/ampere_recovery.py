"""Resource-only fallback after training-fixture parity failure; fixed science retained."""
import argparse,datetime,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from worker import R,O,guards,write,now,sha

def launch(index,uuid):
    guards();host=os.uname().nodename;assert host in ['glacier','rainier']
    lock=(R/f'results/gpu-{host}-{index}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();u,name,mem,util,ecc=[x.strip() for x in q.split(',')]
    assert u==uuid and name=='NVIDIA A40' and int(mem)<100 and int(util)==0 and int(ecc)==0,q
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    recovery=json.loads((O/'resource-recovery.json').read_text());assert sha(Path(__file__))==recovery['launcher_sha256']
    d=O/'features';assert json.loads((d/'exit.json').read_text())['exit_code']==1
    assert json.loads((d/'hardware-parity.json').read_text())['passed'] is False
    assert not list(d.glob('*/seed-*')) and not (O/'result.json').exists(),'No confirmation inference/scoring may have occurred'
    assert not (O/'feature-attempt-l40s').exists()
    d.rename(O/'feature-attempt-l40s');d.mkdir();(d/'launch-claim').mkdir()
    deadline=datetime.datetime.fromisoformat(json.loads((O/'window.json').read_text())['deadline']);seconds=int((deadline-now()).total_seconds())-20;assert seconds>=1800
    write(d/'launch.json',{'started_at':now().isoformat(),'host':host,'gpu_query':q,'timeout_seconds':seconds,'deadline':deadline.isoformat(),'reason':'same frozen feature code and tolerance on original Ampere architecture; L40S fixture failure retained'})
    env=dict(os.environ,WAM_ROOT=str(R),LAYER_RUN=str(R/'results/layers-20260922'),PYTHONHOME=str(R/'assets/python310-runtime'),PYTHONPATH=str(R/'env/lib/python3.10/site-packages'),CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with (d/'run.log').open('x') as log:
        proc=subprocess.run(['timeout','--signal=TERM','--kill-after=15s',str(seconds),str(R/'env/bin/python'),'-u',str(R/'scripts/confirmation/features.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
    write(d/'exit.json',{'finished_at':now().isoformat(),'exit_code':proc.returncode});return proc.returncode

def wait_for_resource():
    guards();assert not (O/'resource-wait-started.json').exists();write(O/'resource-wait-started.json',{'time':now().isoformat()})
    deadline=datetime.datetime.fromisoformat(json.loads((O/'window.json').read_text())['deadline'])-datetime.timedelta(minutes=30)
    while now()<deadline:
        guards();observations=[]
        for host in ['glacier','rainier']:
            cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',f'zifanz4@{host}.csl.illinois.edu']
            q=subprocess.run(cmd+['nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
            observations.append({'host':host,'code':q.returncode,'gpu_query':q.stdout,'stderr':q.stderr})
            if q.returncode:continue
            for line in q.stdout.strip().splitlines():
                index,uuid,name,mem,util,ecc=[x.strip() for x in line.split(',')]
                if name!='NVIDIA A40' or int(mem)>=100 or int(util)!=0 or ecc!='0':continue
                # Remote launcher rechecks availability and takes a cooperative lock.
                write(O/'resource-wait.json',{'stage':'dispatching_ampere','host':host,'index':index,'uuid':uuid,'time':now().isoformat()})
                run=subprocess.run(cmd+[f'python3 {R}/scripts/confirmation/ampere_recovery.py launch --index {int(index)} --uuid {uuid}'],timeout=max(1,(deadline+datetime.timedelta(minutes=30)-now()).total_seconds())+30)
                write(O/'resource-wait-exit.json',{'time':now().isoformat(),'stage':'ampere_attempt_finished','exit_code':run.returncode});return run.returncode
        write(O/'resource-wait.json',{'stage':'waiting_for_idle_A40','time':now().isoformat(),'last_dispatch_by':deadline.isoformat(),'observations':observations,'no_confirmation_scores_seen':True})
        time.sleep(120)
    write(O/'resource-wait-exit.json',{'time':now().isoformat(),'stage':'no_idle_A40_before_dispatch_cutoff','no_scores':True});return 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['wait','launch']);p.add_argument('--index',type=int);p.add_argument('--uuid');a=p.parse_args()
    try:code=wait_for_resource() if a.mode=='wait' else launch(a.index,a.uuid)
    except Exception:
        write(O/'resource-recovery-error.json',{'time':now().isoformat(),'traceback':traceback.format_exc()});code=1
    raise SystemExit(code)
