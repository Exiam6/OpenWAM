"""CPU-only identities, idle probes and full-cap bounds for the joint allocation."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess

R=Path('/home/zifanz4/openwam-runtime')
G=R/'ex4-g_calibratedfactorial_1017'
V=R/'ex5-a_vaebaseline_1018'
PLAN=V/'joint-allocation-plan.json'
ASSIGNMENT=V/'joint-allocation.json'
END=dt.datetime.fromisoformat('2026-10-18T16:59:00+00:00')


def now():return dt.datetime.now(dt.timezone.utc)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def verify(plan_path=PLAN):
    p=read(plan_path)
    assert p['g_registration_sha256']==sha(G/'registration.json')
    assert p['v_registration_sha256']==sha(V/'gpu-registration.json')
    assert p['g_max_seconds']==172800 and p['v_max_seconds']==73800
    assert p['g_job_seconds']==10800 and p['preflight_seconds']==9000
    assert p['training_evaluation_reserved_seconds']==22560
    assert p['allocation_end']==END.isoformat()
    assert p['status']=='registered_resource_only'
    assert p['current_parents'] and len({x['job'] for x in p['current_parents']})==len(p['current_parents'])
    for path,digest in p['files'].items():assert sha(path)==digest,path
    assert str(Path(__file__).resolve()) in p['files']
    return p


def legacy_failed(root=V,before_gpu=False):
    """An expired CPU admission is retained; a prior GPU attempt forbids this route."""
    root=Path(root);e=read(root/'extra-window-admission-expired.json')
    assert e['status']=='resource_admission_expired_before_any_GPU_stage'
    assert e['scientific_gpu_attempt_started'] is False and e['occupied_gpu_seconds']==0
    assert sha(root/'preflight-run.json')==e['preflight_state_sha256']
    assert sha(root/'preflight-schedule-v2.json')==e['scheduler_state_sha256']
    assert sha(root/'scheduled-preflight-v2.log')==e['log_sha256']
    state=read(root/'preflight-run.json');schedule=read(root/'preflight-schedule-v2.json')
    assert state['status']=='failed' and state['stages']==[] and state.get('occupied_seconds',0)==0
    assert state['admitted_samples']==0 and schedule['status']=='failed' and schedule['exit_code']==1
    assert not Path('/proc',str(state['pid'])).exists() and not Path('/proc',str(schedule['pid'])).exists()
    if before_gpu:
        for name in ['vae-cache','training-smoke','deployment-contract','tiny-fit','runs','evaluation','dedicated-preflight-run.json']:
            assert not (root/name).exists(),name
    return e


def chain_window(remaining,current=None):
    assert remaining in [1,2,3]
    current=current or now()
    assert (END-current).total_seconds()>remaining*(10800+60)+60,'Full remaining G chain no longer fits'
    return END


def preflight_window(current=None,future_releases=()):
    current=current or now()
    assert (END-current).total_seconds()>9000+22560,'Complete VAE endpoint no longer fits'
    latest=max([current+dt.timedelta(seconds=9000),*future_releases])
    assert (END-latest).total_seconds()>22560,'Reserved VAE lanes cannot become ready in time'
    return END


def process_matches(pid,fragment):
    path=Path('/proc',str(pid))
    try:return path.stat().st_uid==os.getuid() and fragment in (path/'cmdline').read_bytes().decode(errors='replace')
    except FileNotFoundError:return False


def lock_owned(pid,path):
    info=Path(path).stat();device=f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}'
    for line in Path('/proc/locks').read_text().splitlines():
        fields=line.split()
        if len(fields)>=8 and fields[1]=='FLOCK' and fields[3]=='WRITE' and fields[4]==str(pid) and fields[5]==device:return True
    return False


def capacity(gpus,registration):
    occupied={line.split(',')[0].strip() for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines() if line.strip()}
    result={}
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,memory.used,memory.reserved,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines():
        index,uuid,pci,free,used,reserved,util=[x.strip() for x in line.split(',')];gpu=int(index)
        if gpu not in gpus:continue
        assert gpu in [1,2,3,4] and uuid==registration['gpus'][str(gpu)]
        free,used,reserved,util=map(int,[free,used,reserved,util]);assert min(free,used,reserved,util)>=0
        result[gpu]=dict(gpu=gpu,uuid=uuid,pci=pci,free_mib=free,used_mib=used,driver_reserved_mib=reserved,util=util,idle=uuid not in occupied and used<256 and util==0)
    assert set(result)==set(gpus)
    return result


def check_assignment(plan,require_live=True,check_window=True):
    a=read(ASSIGNMENT);assert a['plan_sha256']==sha(PLAN)
    assert sorted([a['chain_gpu'],*a['vae_gpus']])==[1,2,3,4] and a['chain_gpu'] in [2,3,4]
    first=read(G/'claims'/f"{plan['g_jobs'][0]}.json")
    assert first['status'] in ['running','complete'] and first['dispatch']=='joint_reserved_chain'
    assert first['joint_plan_sha256']==sha(PLAN) and first['gpu']==a['chain_gpu']
    assert first['started']==a['first_chain_started']
    chain_window(3,dt.datetime.fromisoformat(first['started']))
    if require_live:
        assert process_matches(a['coordinator_pid'],'joint_allocation.py'),'Joint coordinator is absent'
        assert Path('/proc',str(a['coordinator_pid']),'stat').read_text().split()[21]==a['coordinator_start_ticks']
        assert lock_owned(a['coordinator_pid'],G/'dedicated-dispatch.lock'),'G reservations are not held'
    releases=[]
    for item in plan['current_parents']:
        if item['gpu'] not in a['vae_gpus']:continue
        claim=read(G/'claims'/f"{item['job']}.json")
        assert claim['status'] in ['running','complete'] and claim['child_pid']==item['pid'] and claim['started']==item['started']
        assert claim['gpu']==item['gpu'] and claim['job']['seconds_limit']==10800
        if claim['status']=='running':assert process_matches(item['pid'],item['job'])
        else:assert read(claim['result'])['complete']
        releases.append(dt.datetime.fromisoformat(item['started'])+dt.timedelta(seconds=10860))
    if check_window:preflight_window(future_releases=releases)
    return a
