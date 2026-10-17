"""Bounded paired expert replay with only the configuration path changed."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-e_plannerpath_1017')
R=ROOT.parent;CODE=Path(__file__).resolve().parent
DEMOS=CODE/'demos'
GPUS={1:('GPU-89770bc8-d310-99d8-b3a4-d2b14c93d1a9','00000000:07:00.0'),
      2:('GPU-5588ad46-6e71-bb2c-8f18-80b2e7a14b5e','00000000:75:00.0')}

def write(path,d):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(path)

def guard():
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,18,17,tzinfo=dt.timezone.utc)
    for p in [ROOT/'paused.json',*[Path(p) for p in json.loads((R/'rae-factorial-20261010/protocol.json').read_text())['pause_paths']]]:
        assert not p.exists(),p

def choose():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    candidates=[]
    for line in rows.splitlines():
        gpu,util,used,total=map(int,line.split(','))
        if gpu not in GPUS or util>85 or total-used<30720:continue
        foreign=False
        for app in apps.splitlines():
            uuid,pid=[s.strip() for s in app.split(',')];p=Path('/proc')/pid
            if uuid==GPUS[gpu][0] and p.exists() and p.stat().st_uid!=os.getuid():foreign=True
        if not foreign:candidates.append((util,gpu))
    return min(candidates)[1] if candidates else None

def main():
    assert json.loads((ROOT/'branch-audit.json').read_text())['passed']
    records=[];child=None;phase='waiting_for_capacity';gpu=None;lock=None;begin=time.monotonic()
    def save():write(ROOT/'queue.json',dict(pid=os.getpid(),phase=phase,gpu=gpu,records=records,time=dt.datetime.now(dt.timezone.utc).isoformat()))
    def term(*_):raise RuntimeError('Replay queue terminated')
    signal.signal(signal.SIGTERM,term);signal.signal(signal.SIGINT,term);save()
    try:
        while gpu is None:
            guard();assert time.monotonic()-begin<1800
            gpu=choose()
            if gpu is None:time.sleep(10)
        lock=(R/f'gpu-{gpu}-ex4e.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(HOME='/home/zifanz4',USER='zifanz4',PATH=f'{R}/benchmark-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin',
            CUDA_VISIBLE_DEVICES=GPUS[gpu][0],EXPECTED_RENDER_PCI=GPUS[gpu][1],
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
            CUDA_HOME=str(R/'cuda-toolkit'),CPATH=str(R/'cuda-toolkit/targets/x86_64-linux/include'),
            TORCH_CUDA_ARCH_LIST='12.0',TMPDIR=str(R/'tmp'),MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2',
            TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions-sm120'),TRITON_CACHE_DIR=str(R/'cache/triton-sm120'),
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',
            ROBOTWIN_PATH=str(R/'assets-source/benchmarks/RoboTwin'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1',
            WAM_ROOT=str(R/'assets-source'),LD_LIBRARY_PATH=str(R/'benchmark-env/lib/python3.10/site-packages/sapien/oidn_library'),
            REPLAY_HELPERS=str(CODE),REPLAY_PROPRIO='1',REPLAY_FRAMES='1')
        write(ROOT/'environment.json',env)
        primary='handover_block-seed4101100'
        demos=[primary,*sorted(p.name for p in DEMOS.iterdir() if p.is_dir() and p.name!=primary)]
        assert len(demos)==9
        active=time.monotonic();outputs={}
        for demo in demos:
            for condition in ['old','fixed']:
                guard();assert time.monotonic()-active<3600
                out=ROOT/'replay'/condition/demo;assert not out.exists()
                out.parent.mkdir(parents=True,exist_ok=True)
                cfg=R/'migration-20260927/robot-configs' if condition=='old' else ROOT/'robot-configs/aloha-agilex'
                runenv=dict(env,ROBOT_CONFIG_ROOT=str(cfg),ROBOTWIN_RUNTIME_ROOT=str(ROOT/'simulator-runtime'/condition))
                cmd=[str(R/'benchmark-env/bin/python'),'-u',str(CODE/'replay_expert_ee.py'),str(out),str(DEMOS/demo)]
                logfile=ROOT/f'{condition}-{demo}.log'
                with logfile.open('x') as log:
                    child=subprocess.Popen(cmd,env=runenv,cwd='/home/zifanz4/openwam-rae-research',stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                rec=dict(condition=condition,demo=demo,pid=child.pid,command=cmd,config_root=str(cfg),started=dt.datetime.now(dt.timezone.utc).isoformat())
                records.append(rec);phase='running';save();cell_start=time.monotonic()
                while child.poll() is None:
                    guard();assert time.monotonic()-active<3600 and time.monotonic()-cell_start<600
                    time.sleep(5)
                rec.update(exit_code=child.returncode,finished=dt.datetime.now(dt.timezone.utc).isoformat());save()
                assert child.returncode==0,rec
                row=json.loads((out/f'{demo}.json').read_text())
                assert 'error' not in row,row
                assert row['initial_proprio_max_abs_diff']<=1e-6,row
                assert row['left_fail']==0 and row['right_fail']==0,row
                outputs[(condition,demo)]=row;rec['result']=row;save();child=None
            if demo==primary:
                assert not outputs[('old',demo)]['success'],'Old path must reproduce known failure'
                assert outputs[('fixed',demo)]['success'],'Fixed-path L1 did not restore handover; diagnose before full set'
                write(ROOT/'L1-gate.json',dict(passed=True,old=outputs[('old',demo)],fixed=outputs[('fixed',demo)]))
        phase='complete'
        write(ROOT/'paired-summary.json',dict(complete=True,records=records))
    except BaseException as exc:
        phase='stopped';records.append(dict(error=repr(exc)));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save()
        if lock is not None:lock.close()

if __name__=='__main__':main()
