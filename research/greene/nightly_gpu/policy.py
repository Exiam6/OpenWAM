"""Nightly admission uses a complete recent observation history, never a snapshot."""
import datetime as dt
import math
from zoneinfo import ZoneInfo


def window(now):
    local = now.astimezone(ZoneInfo('America/Chicago'))
    start = local.replace(hour=1,minute=0,second=0,microsecond=0,fold=0)
    end = local.replace(hour=9,minute=30,second=0,microsecond=0,fold=0)
    utc = dt.timezone.utc
    return start.astimezone(utc), end.astimezone(utc), start.astimezone(utc)<=now<end.astimezone(utc)


def advance(previous, sample, now, policy_hash, max_gap=25):
    """A busy, missing, stale, changed-UUID or gapped sample resets continuity."""
    result = {}
    for gpu in sample.get('gpus', []):
        key = str(gpu['index'])
        old = previous.get(key,{})
        stamp = now.timestamp()
        idle = gpu['idle'] is True and not gpu['processes'] and gpu['utilization']==0 and gpu['used_mib']<256
        continuous = (idle and old.get('idle') and old.get('uuid')==gpu['uuid']
                      and old.get('policy_hash')==policy_hash and 0<stamp-old.get('last',0)<=max_gap)
        since = old['since'] if continuous else (stamp if idle else None)
        seconds = stamp-since if since is not None else 0
        result[key] = dict(gpu, idle=idle, since=since, last=stamp, idle_seconds=seconds,
                           policy_hash=policy_hash, qualified=idle and seconds>=1800)
    return result


def dedicated(policy, host, index):
    return ((host, index) in [('gpu01', 1), ('gpu01', 2)]
            and [host, index] in policy.get('always_available', []))


def eligible(row, now, max_seconds, policy=None, host=None):
    _, end, inside = window(now)
    priority = dedicated(policy or {}, host, row.get('index'))
    return ((inside or priority) and row.get('qualified') is True and row.get('idle') is True
            and not row['processes'] and row['utilization']==0 and row['used_mib']<256
            and 0<=now.timestamp()-row['last']<=25
            and math.isfinite(max_seconds) and max_seconds>0
            and (priority or (end-now).total_seconds()>max_seconds+120))


def validate_job(job):
    import re
    assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',job['id'])
    assert job['status']=='ready' and job['scientific_identity'].strip()
    assert isinstance(job['max_seconds'],int) and 60<job['max_seconds']<=30480
    assert job['authorization_evidence'] and len(job['authorization_sha256'])==64
    assert job['host_commands'] and set(job['host_commands'])<={'gpu01','gpu02'}
    for command in job['host_commands'].values():
        for name in ['argv','admission_argv']:
            assert isinstance(command[name],list) and command[name] and all(isinstance(x,str) for x in command[name])
            assert command[name][0].startswith('/')
        assert command['cwd'].startswith('/') and command['input_sha256'] and command['outputs_absent']
    return job
