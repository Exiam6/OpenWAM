"""One finite resource wait for an existing owned controller; no experiment replay."""
import argparse, datetime, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
P = Path('/home/zifanz4/openwam-experiments')
S = P / 'studies/endtoend-20260923'
E = Path('/data02/zifanz4/openwam-experiments/endtoend-20260923')
PLAN = S / 'evaluation-resource-wait-20260924.json'

def now():
    return datetime.datetime.now().astimezone()

def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def identity(pid):
    p = Path('/proc') / str(pid)
    return {'pid': pid, 'uid': p.stat().st_uid,
            'argv': [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x],
            'start_ticks': (p/'stat').read_text().rsplit(')', 1)[1].split()[19]}

def idle(snapshot):
    rows = snapshot['gpu']
    return set(rows) == {str(i) for i in range(1, 7)} and all(
        int(r[2]) < 100 and int(r[3]) == 0 and int(r[4]) == 0
        and r[1] not in snapshot['compute_uuids'] for r in rows.values())

def sample():
    code = """import subprocess,json
a=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
b=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True)
rows=[list(map(str.strip,x.split(','))) for x in a.splitlines()]
print(json.dumps({'gpu':{r[0]:r for r in rows if r[0] in [str(i) for i in range(1,7)]},'compute_uuids':b.splitlines()}))
"""
    r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
                        'zifanz4@cm009.csl.illinois.edu','python3','-'],
                       input=code, text=True, capture_output=True, timeout=15)
    if r.returncode:
        raise RuntimeError('Resource SSH query failed: ' + r.stderr[-400:])
    return json.loads(r.stdout)

def verify(plan):
    assert identity(plan['controller']['pid']) == plan['controller'], 'Controller identity changed'
    assert plan['controller']['uid'] == os.getuid()
    assert plan['controller']['argv'] == ['python3', str(P/'scripts/endtoend/pipeline_controller.py')]
    assert now() < datetime.datetime.fromisoformat(plan['deadline'])
    for path in plan['pause_paths']:
        assert not Path(path).exists(), 'User pause: ' + path
    for path, expected in plan['sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    state = json.loads((E/'pipeline/progress.json').read_text())
    assert state['stage'] == 'waiting_for_all_training_collection_smoke' and not state['all_ready'], state
    assert not (E/'pipeline/exit.json').exists()

def selftest():
    good = {'gpu':{str(i):[str(i),'GPU-'+str(i),'1','0','0'] for i in range(1,7)}, 'compute_uuids':[]}
    assert idle(good)
    for col,val in [(2,'100'),(3,'1'),(4,'1')]:
        bad=json.loads(json.dumps(good));bad['gpu']['1'][col]=val;assert not idle(bad)
    bad=json.loads(json.dumps(good));bad['compute_uuids']=['GPU-4'];assert not idle(bad)
    bad=json.loads(json.dumps(good));del bad['gpu']['6'];assert not idle(bad)
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        before=identity(child.pid);fd=os.pidfd_open(child.pid)
        try:
            signal.pidfd_send_signal(fd,signal.SIGSTOP)
            for _ in range(100):
                if Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[0]=='T':break
                time.sleep(.01)
            else:raise AssertionError('Owned test process did not stop')
            assert identity(child.pid)==before
            signal.pidfd_send_signal(fd,signal.SIGCONT)
        finally:os.close(fd)
    finally:
        child.terminate();child.wait(timeout=5)
    print('PASS: idle/ECC/process predicates, missing GPU rejection, pidfd stop/resume of own test child')

def main(execute):
    plan=json.loads(PLAN.read_text());verify(plan);out=Path(plan['output'])
    assert not out.exists(), 'One-shot wait already executed'
    initial=sample()
    if not execute:
        print(json.dumps({'checks_passed':True,'resource_ready_now':idle(initial),'snapshot':initial}))
        return
    out.mkdir()
    write(out/'started.json',{'time':now().isoformat(),'plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'controller':plan['controller']})
    fd=os.pidfd_open(plan['controller']['pid']);stopped=False;status='error';count=0;rc=1
    def interrupted(signum,frame):
        raise SystemExit('Resource wait interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:
        verify(plan)
        signal.pidfd_send_signal(fd,signal.SIGSTOP);stopped=True
        time.sleep(.1)
        verify(plan)  # Must still be in the not-ready dependency loop, never intercept an active evaluation.
        while True:
            if now() >= datetime.datetime.fromisoformat(plan['deadline']):
                status='original_dependency_deadline_reached';break
            if any(Path(p).exists() for p in plan['pause_paths']):
                status='user_pause_detected';break
            assert identity(plan['controller']['pid'])==plan['controller']
            snapshot=sample();count=count+1 if idle(snapshot) else 0
            write(out/'progress.json',{'time':now().isoformat(),'status':'waiting_for_original_six_slots','controller_held':True,'consecutive_idle':count,'snapshot':snapshot,'deadline':plan['deadline']})
            if count>=3:
                status='original_slots_idle_controller_released';rc=0;break
            time.sleep(5 if count else 30)
    except BaseException as exc:
        write(out/'error.json',{'time':now().isoformat(),'error':repr(exc)})
        raise
    finally:
        if stopped:
            try:signal.pidfd_send_signal(fd,signal.SIGCONT)
            except ProcessLookupError:pass
        os.close(fd)
        write(out/'exit.json',{'time':now().isoformat(),'returncode':rc,'status':status,'controller_resumed_if_alive':stopped,'original_deadlines_unchanged':True})
    sys.exit(rc)

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--selftest',action='store_true');args=ap.parse_args()
    if args.selftest:selftest()
    else:main(args.execute)
