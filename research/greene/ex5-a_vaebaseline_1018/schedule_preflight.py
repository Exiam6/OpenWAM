"""CPU-only one-shot wait for the registered window, then invoke frozen preflight."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from build_cache import sha


def main(registration,gpu):
    path=Path(registration).resolve();root=path.parent;p=json.loads(path.read_text());digest=sha(path)
    assert p['gpu_launch_permitted'] and p['preflight_launch_permitted'] and gpu in [0,5,6,7]
    start=dt.datetime.fromisoformat(p['extra_window_start'])
    lock=(root/'preflight-schedule.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=root/'preflight-schedule.json'
    state=dict(pid=os.getpid(),status='waiting_authorized_window',gpu=gpu,registration_sha256=digest,
        window_start=start.isoformat(),created=dt.datetime.now(dt.timezone.utc).isoformat(),child_pid=None)
    with state_path.open('x') as f:json.dump(state,f,indent=2)
    def save():
        state['updated']=dt.datetime.now(dt.timezone.utc).isoformat()
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    try:
        while True:
            assert sha(path)==digest,'Registration changed during wait'
            for pause in p['pause_paths']:assert not Path(pause).exists(),pause
            now=dt.datetime.now(dt.timezone.utc)
            assert now<dt.datetime.fromisoformat(p['deadline'])
            if now>=start:break
            save();time.sleep(min(30,max(.1,(start-now).total_seconds())))
        assert not (root/'preflight-run.json').exists(),'Preserve any previous preflight attempt'
        script=Path(p['source_snapshot'])/'run_preflight.py'
        assert sha(script)==p['launcher_files'][str(script)]
        # The frozen parent independently checks all assets, window and real GPU occupancy.
        command=[str(root.parent/'policy-env/bin/python'),'-u',str(script),'--gpu',str(gpu)]
        state.update(status='launching_preflight',command=command);save()
        with (root/'scheduled-preflight.log').open('x') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                cwd='/home/zifanz4/openwam-rae-research',start_new_session=True)
        state.update(status='preflight_running',child_pid=child.pid);save()
        code=child.wait();state.update(status='complete' if code==0 else 'failed',exit_code=code);save()
    except BaseException as error:
        state.update(status='failed',error=repr(error));save();raise
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--registration',required=True);parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();main(args.registration,args.gpu)
