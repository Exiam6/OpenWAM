"""Continuously inventory16GPUs; claim each ready experiment at most once."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from policy import advance,eligible,validate_job,window
from worker import sha,write

ROOT=Path('/home/zifanz4/openwam-runtime/nightly-gpu')
SOURCE=Path(__file__).parent


def command(policy,host,argv,**kwargs):
    target=policy['hosts'][host]['ssh']
    args=argv if target is None else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',target,shlex.join(argv)]
    return subprocess.run(args,capture_output=True,text=True,timeout=kwargs.pop('timeout',20),**kwargs)


def sample(policy,host):
    result=command(policy,host,['/usr/bin/python3',str(SOURCE/'probe.py')])
    if result.returncode:raise RuntimeError(result.stderr.strip())
    data=json.loads(result.stdout)
    assert data['hostname']==policy['hosts'][host]['hostname']
    assert {str(r['index']):r['uuid'] for r in data['gpus']}==policy['hosts'][host]['gpus']
    return data


def refresh_claim(policy,path):
    claim=json.loads(path.read_text())
    if claim['status'] not in ['reserved','submitted','running']:return claim
    # The remote worker's result, not an SSH session, is authoritative.
    target=ROOT/'runs'/claim['id']/'result.json'
    query=['/usr/bin/python3','-c','import pathlib,sys; p=pathlib.Path(sys.argv[1]); print(p.read_text() if p.exists() else "null")',str(target)]
    result=command(policy,claim['host'],query)
    if result.returncode:return claim
    state=json.loads(result.stdout)
    if state is not None:
        assert state['id']==claim['id'] and state['gpu_uuid']==claim['gpu_uuid']
        claim.update(status=state['status'],worker_result=state)
        if state['status'] in ['validating','running']:claim['status']='running'
        write(path,claim)
    if claim['status'] in ['submitted','running']:
        unit=claim.get('unit')
        if unit:
            check=command(policy,claim['host'],['systemctl','--user','show',unit,'--property=LoadState,ActiveState'])
            values=dict(line.split('=',1) for line in check.stdout.splitlines() if '=' in line)
            age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(claim['created'])).total_seconds()
            if age>30 and values.get('ActiveState') in ['inactive','failed']:
                claim.update(status='failed',error='Owned unit exited without a successful worker result',unit_state=values)
                write(path,claim)
    return claim


def dispatch(policy,job,host,gpu,row):
    now=dt.datetime.now(dt.timezone.utc)
    assert eligible(row,now,job['max_seconds'])
    _,end,_=window(now)
    claim=dict(id=job['id'],scientific_identity=job['scientific_identity'],status='reserved',host=host,
               gpu_index=gpu,gpu_uuid=row['uuid'],created=now.isoformat(),idle_evidence=row,
               policy_sha256=sha(ROOT/'policy.json'),job_sha256=sha(ROOT/'jobs'/f"{job['id']}.json"))
    path=ROOT/'claims'/f"{job['id']}.json"
    with path.open('x') as handle:json.dump(claim,handle,indent=2)
    try:
        current=next(g for g in sample(policy,host)['gpus'] if g['index']==gpu)
        assert current['idle'] and current['uuid']==row['uuid'],'Capacity changed before dispatch'
        spec=ROOT/'incoming'/f"{job['id']}.json"
        writer='import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); f=p.open("x"); f.write(sys.stdin.read()); f.close()'
        result=command(policy,host,['/usr/bin/python3','-c',writer,str(spec)],input=json.dumps(job))
        assert result.returncode==0,result.stderr
        unit='openwam-nightly-job-'+job['id']
        argv=['systemd-run','--user','--quiet','--collect','--unit',unit,
              '--property=KillMode=control-group','--property=TimeoutStopSec=15',
              f"--property=RuntimeMaxSec={job['max_seconds']-15}",
              '/usr/bin/python3',str(SOURCE/'worker.py'),'--spec',str(spec),'--host',host,
              '--gpu-uuid',row['uuid'],'--allocation-end',(end-dt.timedelta(seconds=60)).isoformat(),'--runtime',str(ROOT)]
        result=command(policy,host,argv)
        assert result.returncode==0,result.stderr
        claim.update(status='submitted',unit=unit,command=argv)
    except BaseException as error:
        # An uncertain submit is retained as claimed; never launch it a second time.
        claim.update(status='submission_uncertain',error=repr(error))
    write(path,claim)
    return claim


def main():
    for name in ['jobs','claims','observations']: (ROOT/name).mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        policy=json.loads((ROOT/'policy.json').read_text());policy_hash=sha(ROOT/'policy.json')
        assert policy['timezone']=='America/Chicago' and policy['start']=='01:00' and policy['end']=='09:30'
        assert policy['idle_seconds']==1800 and policy['poll_seconds']==10
        manifest=json.loads((ROOT/'source-manifest.json').read_text())
        for path,digest in manifest['files'].items():assert sha(path)==digest,path
        previous_path=ROOT/'latest.json';previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
        histories=previous.get('hosts',{}) if previous.get('policy_sha256')==policy_hash else {}
        rotation=0
        with ThreadPoolExecutor(max_workers=2) as pool:
            while True:
                tick=time.monotonic();now=dt.datetime.now(dt.timezone.utc)
                # Re-reading guards against accidental live policy edits; a new version requires a restart.
                assert sha(ROOT/'policy.json')==policy_hash
                futures={h:pool.submit(sample,policy,h) for h in policy['hosts']}
                errors={}
                for host,future in futures.items():
                    try:
                        sampled=future.result()
                        sampled_at=dt.datetime.fromisoformat(sampled['time'])
                        received=dt.datetime.now(dt.timezone.utc)
                        assert abs((received-sampled_at).total_seconds())<15,'Stale remote clock or sample'
                        histories[host]=advance(histories.get(host,{}),sampled,received,policy_hash)
                    except Exception as error:
                        histories[host]={};errors[host]=repr(error)
                now=dt.datetime.now(dt.timezone.utc);_,end,inside=window(now)
                claims=[]
                for path in sorted((ROOT/'claims').glob('*.json')):
                    try:claims.append(refresh_claim(policy,path))
                    except Exception as error:errors[str(path)]=repr(error)
                claimed_ids={c['id'] for c in claims};claimed_science={c['scientific_identity'] for c in claims}
                protected=json.loads((ROOT/'protected-identities.json').read_text())
                claimed_science.update(protected['scientific_identities'])
                busy={(c['host'],c['gpu_index']) for c in claims if c['status'] not in ['complete','failed']}
                candidates=[]
                for host,gpus in histories.items():
                    for index,row in gpus.items():
                        if (host,int(index)) not in busy:candidates.append((host,int(index),row))
                candidates.sort(key=lambda x:(0 if x[:2] in [('gpu01',1),('gpu01',2)] else 1,
                                              (list(policy['hosts']).index(x[0])*8+x[1]-rotation)%16))
                rotation=(rotation+1)%16
                pending=[]
                for path in sorted((ROOT/'jobs').glob('*.json')):
                    try:
                        job=validate_job(json.loads(path.read_text()));assert path.stem==job['id']
                        if job['id'] in claimed_ids or job['scientific_identity'] in claimed_science:continue
                        assert sha(job['authorization_evidence'])==job['authorization_sha256']
                        pending.append(job['id'])
                        for host,index,row in list(candidates):
                            if host in job['host_commands'] and eligible(row,dt.datetime.now(dt.timezone.utc),job['max_seconds']):
                                claim=dispatch(policy,job,host,index,row);claims.append(claim)
                                claimed_ids.add(job['id']);claimed_science.add(job['scientific_identity'])
                                candidates.remove((host,index,row));pending.remove(job['id']);break
                    except Exception as error:errors[str(path)]=repr(error)
                state=dict(time=now.isoformat(),pid=os.getpid(),policy_sha256=policy_hash,inside_window=inside,
                           window_end=end.isoformat(),hosts=histories,errors=errors,claims=claims,pending_jobs=pending,
                           registered_jobs=len(list((ROOT/'jobs').glob('*.json'))),
                           note='Eligibility requires observed30min continuity; no claim of activity between10s samples. Existing dedicated experiments are observed but never restarted or terminated by this controller.')
                write(previous_path,state)
                with gzip.open(ROOT/'observations'/f'{now.date().isoformat()}.jsonl.gz','at') as handle:
                    handle.write(json.dumps(state,separators=(',',':'))+'\n')
                time.sleep(max(.1,policy['poll_seconds']-(time.monotonic()-tick)))


if __name__=='__main__':main()
