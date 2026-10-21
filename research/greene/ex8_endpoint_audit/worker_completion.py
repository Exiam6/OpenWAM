"""Independent terminal-worker evidence; never mutate dispatcher state."""
import datetime as dt
import hashlib
import json
import pathlib


def terminal_state(job, claim, worker, row, job_sha, spec_sha):
    if claim is None or worker is None:
        return 'pending'
    for field in ('id', 'scientific_identity'):
        assert job[field] == claim[field] == worker[field], field
    assert claim['job_sha256'] == job_sha, 'Job digest mismatch'
    assert worker['spec_sha256'] == spec_sha, 'Worker spec digest mismatch'
    assert claim['host'] == worker['host'] == row['host']
    assert claim['gpu_uuid'] == worker['gpu_uuid'] == row['gpu_uuid']
    assert claim['gpu_index'] == worker['gpu_index']
    assert worker['command'] == job['host_commands'][worker['host']]['argv']
    if claim['status'] == 'failed' or worker['status'] == 'failed':
        return 'failed'
    if claim['status'] != 'complete' or worker['status'] != 'complete':
        return 'pending'
    assert worker['exit_code'] == 0, 'Nonzero terminal exit'
    assert claim['worker_result'] == worker, 'Final claim/worker disagreement'
    idle = claim['idle_evidence']
    assert idle['idle'] and idle['qualified'] and idle['idle_seconds'] >= 1800
    assert idle['uuid'] == worker['gpu_uuid'] and idle['index'] == worker['gpu_index']
    assert idle['policy_hash'] == claim['policy_sha256']
    assert dt.datetime.fromisoformat(worker['started']) <= dt.datetime.fromisoformat(row['started'])
    assert dt.datetime.fromisoformat(row['finished']) <= dt.datetime.fromisoformat(worker['finished'])
    assert worker['occupied_parent_seconds'] <= job['max_seconds'], 'Worker cap exceeded'
    return 'complete'


def confirm_rows(rows, runtime):
    root = runtime/'nightly-gpu'
    jobs = [(f, json.loads(f.read_text())) for f in sorted((root/'jobs').glob('ex8d-full-*.json'))]
    confirmed, checks = [], []
    def digest(f):
        return hashlib.sha256(f.read_bytes()).hexdigest()
    for row in rows:
        stage = runtime/('ex8-e_rowrecovery_1021' if row['key']=='demo_clean-adjust_bottle' else 'ex8-d_nativefull_1021')
        out = stage/'runs'/row['key']
        matches = [(f,j) for f,j in jobs if j['authorization_evidence']==str(stage/'protocol.json')
                   and all(str(out) in c['outputs_absent'] for c in j['host_commands'].values())]
        assert len(matches)==1, 'Missing/ambiguous registered job for '+row['key']
        jobfile, job = matches[0]
        assert digest(stage/'protocol.json') == job['authorization_sha256']
        claimfile = root/'claims'/(job['id']+'.json')
        workerfile = root/'runs'/job['id']/'result.json'
        specfile = root/'incoming'/(job['id']+'.json')
        files = [jobfile, out/'result.json']
        status = 'pending'
        if claimfile.exists() and workerfile.exists() and specfile.exists():
            claim, worker = json.loads(claimfile.read_text()), json.loads(workerfile.read_text())
            assert json.loads(specfile.read_text()) == job, 'Incoming spec changed'
            status = terminal_state(job, claim, worker, json.loads((out/'result.json').read_text()), digest(jobfile), digest(specfile))
            files.extend([claimfile, workerfile, specfile])
        checks.append(dict(key=row['key'], job_id=job['id'], status=status,
                           evidence_sha256={str(f):digest(f) for f in files}))
        if status == 'complete':
            confirmed.append(row)
    return confirmed, checks


def select_completed(result, confirmed, checks):
    """Strip any artifact-only headline until all supervising workers finish."""
    result['artifact_completed_rows'] = result['completed_rows']
    result['dispatcher_completion_checks'] = checks
    result['dispatcher_failed_rows'] = [x['key'] for x in checks if x['status']=='failed']
    result['pending'] = sorted(set(result['pending']) | {x['key'] for x in checks if x['status']=='pending'})
    result['per_task'] = confirmed
    result['completed_rows'] = len(confirmed)
    result['scored_episodes'] = 100*len(confirmed)
    result['full_endpoint_complete'] = len(confirmed)==100
    for field in ('rates_pct', 'gaps_pp', 'targets_pct', 'task_bootstrap_95pct', 'conclusion'):
        result.pop(field, None)
    result['practical_agreement_passed'] = False
    result['exact_published_numbers_matched'] = False
    included = {x['key'] for x in confirmed}
    for check in result.get('independent_native_outcome_checks', []):
        check['included_in_endpoint'] = check['key'] in included
    return result
