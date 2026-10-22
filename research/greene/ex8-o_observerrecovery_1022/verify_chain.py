"""Additional read-only acceptance check; never imports native GPU libraries."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

from verify_raw import audit as raw_audit, F, N, J, check


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    result=dt.datetime.fromisoformat(value)
    check(result.tzinfo is not None,'Naive terminal timestamp')
    return result


def verify_chain(job,claim,worker,spec,*,job_sha,spec_sha,protocol_sha,expected_id):
    check(job==spec,'Incoming spec differs from registered job')
    check(job['id']==claim['id']==worker['id']==expected_id,'Job identity mismatch')
    check(bool(job['scientific_identity']) and job['scientific_identity']==claim['scientific_identity']==worker['scientific_identity'],'Scientific identity mismatch')
    check(claim['job_sha256']==job_sha,'Registered job digest mismatch')
    check(worker['spec_sha256']==spec_sha,'Executed spec digest mismatch')
    check(job['authorization_sha256']==protocol_sha,'Authorization digest mismatch')
    check(worker['status']==claim['status']=='complete','Nonterminal or failed dispatcher evidence')
    check(type(worker['exit_code']) is int and worker['exit_code']==0,'Invalid terminal exit')
    check(claim['worker_result']==worker,'Final claim/worker disagreement')
    check(claim['host']==worker['host'] and claim['gpu_uuid']==worker['gpu_uuid'],'Allocation mismatch')
    check(type(worker['gpu_index']) is int and type(claim['gpu_index']) is int and worker['gpu_index']>=0 and claim['gpu_index']==worker['gpu_index'],'Allocation index mismatch')
    check(worker['command']==job['host_commands'][worker['host']]['argv'],'Executed command mismatch')
    cap=job['max_seconds'];elapsed=worker['occupied_parent_seconds']
    check(type(cap) is int and cap>0,'Invalid registered cap')
    check(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=cap,'Invalid or exceeded occupied duration')
    check(stamp(claim['created'])<=stamp(worker['started'])<=stamp(worker['finished']),'Invalid terminal timestamp order')
    idle=claim['idle_evidence'];duration=idle['idle_seconds']
    check(idle['idle'] is True and idle['qualified'] is True,'Missing idle admission')
    check(type(duration) in (int,float) and math.isfinite(duration) and duration>=1800,'Insufficient observed idle')
    check(type(idle['since']) in (int,float) and type(idle['last']) in (int,float) and math.isfinite(idle['since']) and math.isfinite(idle['last']),'Invalid idle interval')
    check(abs(idle['last']-idle['since']-duration)<.01,'Idle interval contradicts duration')
    check(idle['uuid']==worker['gpu_uuid'] and idle['index']==worker['gpu_index'],'Idle allocation mismatch')
    check(idle['processes']==[],'Occupied admission evidence')
    check(bool(claim['policy_sha256']) and idle['policy_hash']==claim['policy_sha256'],'Admission policy mismatch')
    return dict(dispatcher_chain_validated=True,registered_command_verified=True,terminal_claim_worker_agreement=True,idle30min_evidence_verified=True)


def audit():
    out=dict(GPU_diagnostic_validated=False,native_L2_L3_passed=False,full_release_allowed=False,full_paper_reproduced=False,audit_kind='INDEPENDENT_RAW_AND_DISPATCHER_CHAIN')
    paths=[N/'jobs'/(J+'.json'),N/'claims'/(J+'.json'),N/'runs'/J/'result.json',N/'incoming'/(J+'.json'),F/'protocol.json']
    required=paths+[F/'result.json',F/'observer-config.json']
    missing=[str(p) for p in required if not p.exists()]
    if missing:return dict(out,assessment='pending evidence',missing=missing)
    try:
        job,claim,worker,spec,protocol=[json.loads(p.read_text()) for p in paths]
        check(job['authorization_evidence']==str(F/'protocol.json'),'Wrong authorization path')
        check(stamp(worker['finished'])<=stamp(protocol['deadline']),'Stage deadline exceeded')
        proof=verify_chain(job,claim,worker,spec,job_sha=sha(paths[0]),spec_sha=sha(paths[3]),protocol_sha=sha(paths[4]),expected_id=J)
        baseline=raw_audit()
        check(baseline['GPU_diagnostic_validated'],'Raw native audit did not validate: '+str(baseline))
        return dict(out,**proof,GPU_diagnostic_validated=True,assessment='Native prefix only; separate native scene L2/L3 required',raw_audit=baseline,evidence_sha256={str(p):sha(p) for p in required},audit_source_sha256=sha(Path(__file__)))
    except Exception as exc:return dict(out,assessment='failed closed',error=repr(exc))

if __name__=='__main__':print(json.dumps(audit(),indent=2))
