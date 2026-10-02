"""Fail-closed Git-CAS scheduling. Dedicated checkout; never rebase a claim."""
import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

TERMINAL = {'complete', 'failed', 'interrupted'}


def require(value, message):
    if not value:
        raise ValueError(message)


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def stamp(value):
    return value.isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parse_time(value):
    result = dt.datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'Timestamp must include UTC offset')
    return result


def fresh(record, now):
    age = (now - parse_time(record['updated_at'])).total_seconds()
    require(-30 <= age <= 300, 'Resource report stale or clock skew; refresh before admission')


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def active_shards(state, actor):
    return [sh for unit in state['units'].values() for sh in unit['shards'].values()
            if sh.get('cluster') == actor and sh['state'] in {'claimed', 'running'}]


def proof(evidence, *fields):
    require(is_sha256(evidence.get('receipt_sha256')),
            'Supply SHA256 of the locally retained verification receipt')
    require(all(evidence.get(field) is True for field in fields), 'Missing affirmative stop/admission evidence')


def resource(state, actor, unit, evidence, now):
    rid = evidence.get('allocation_id')
    r = state['resources'].get(actor, {}).get(rid)
    require(r is not None, 'Actual allocation not registered')
    fresh(r, now)
    require(r['assets_verified'] and r['smoke_passed'], 'Assets or smoke gate missing')
    require(unit['manifest_sha256'] in r['verified_manifests'], 'This exact unit manifest is not verified')
    require(unit['hardware'] in (None, r['hardware']), 'Hardware class must remain fixed within seed')
    require(r['runner_admission_verified'], 'Runner must enforce this registry before every cell')
    return rid, r


def owned(state, req):
    unit = state['units'][req['unit']]
    require(unit['owner'] == req['actor'], 'Unit belongs to another cluster')
    return unit


def shard_token(unit, req):
    shard = unit['shards'][req['shard']]
    require(unit['mode'] == 'managed', 'Legacy admission has not been disabled')
    require(shard.get('token') == req.get('token') and bool(req.get('token')), 'Stale or missing fencing token')
    require(shard.get('worker') == req.get('worker'), 'Another worker owns this shard')
    require(shard.get('epoch') == unit['epoch'], 'Unit epoch changed')
    return shard


def apply(state, req, now=None):
    """Pure transition; a caller may execute work only after its Git push succeeds."""
    now = now or utcnow()
    s = copy.deepcopy(state)
    action, actor = req['action'], req['actor']
    e = req.get('evidence', {})
    require(actor in s['clusters'], 'Unknown cluster')
    if action in {'claim', 'start-cell', 'transfer'}:
        require(now < parse_time(s['new_work_deadline']), 'New work deadline passed; ownership does not expire')
    require(not req.get('operation_id') or not any(x['id'] == req['operation_id'] for x in s['events']),
            'Operation already recorded; inspect state, never replay a launch')
    result = {}
    if action == 'advertise':
        require(e.get('allocation_id') and e.get('hardware'), 'Actual allocation ID and hardware required')
        proof(e, 'allocated', 'qos_verified')
        slots = e.get('slots', 0)
        require(isinstance(slots, int) and 0 < slots <= e.get('verified_gpu_limit', 0), 'Exceeds verified GPU/QOS limit')
        require(slots <= s['clusters'][actor]['max_gpu_slots'], 'Exceeds configured cluster GPU cap')
        workers = e.get('workers_per_gpu', 1)
        require(workers in (1, 2), 'Only one or two workers per GPU supported')
        if workers == 2:
            proof(e, 'parallel_smoke_passed', 'parity_passed', 'no_oom')
            require(e.get('throughput_gain', 0) >= 1.15, 'Two workers must improve throughput by >=15%')
            require(e.get('peak_memory_fraction', 1) <= .8, 'Keep >=20% GPU memory headroom')
        manifests = e.get('verified_manifests', [])
        require(isinstance(manifests, list), 'verified_manifests must be a list')
        active = [sh for sh in active_shards(s, actor) if sh['allocation_id'] == e['allocation_id']]
        old = s['resources'].get(actor, {}).get(e['allocation_id'])
        if active:
            require(old['hardware'] == e['hardware'], 'Cannot change active allocation hardware')
            require(len(active) <= slots * workers, 'Cannot shrink below active worker count')
        s['resources'].setdefault(actor, {})[e['allocation_id']] = dict(
            updated_at=stamp(now), hardware=e['hardware'], slots=slots, workers_per_gpu=workers,
            assets_verified=e.get('assets_verified') is True, smoke_passed=e.get('smoke_passed') is True,
            runner_admission_verified=e.get('runner_admission_verified') is True,
            verified_manifests=manifests, receipt_sha256=e['receipt_sha256'])
    else:
        u = owned(s, req)
        if action == 'adopt':
            require(u['mode'] == 'legacy', 'Unit already managed')
            proof(e, 'legacy_admission_disabled', 'no_queued_or_running_cells', 'gated_runner_verified', 'inventory_verified')
            previous = {k:v for sh in u['shards'].values() for k,v in sh['cells'].items()}
            inventory = e.get('cell_states', {})
            require(set(inventory) == set(previous), 'Adoption must cover exact cell set')
            require(all(v in {'pending'} | TERMINAL for v in inventory.values()), 'Cannot adopt a live cell')
            for cid, old in previous.items():
                require(old['state'] not in TERMINAL or inventory[cid] == old['state'], 'Terminal outcome is immutable')
                require(old['state'] != 'running' or inventory[cid] in TERMINAL, 'An attempted cell cannot become pending')
            for sh in u['shards'].values():
                for cid, cell in sh['cells'].items():
                    cell['state'] = inventory[cid]
                    cell['attempted'] = inventory[cid] != 'pending'
                sh['state'] = 'done' if all(c['state'] in TERMINAL for c in sh['cells'].values()) else 'pending'
            u.update(mode='managed', epoch=u['epoch']+1, admission_receipt=e['receipt_sha256'])
        elif action == 'transfer':
            require(u['mode'] == 'managed', 'Cannot transfer from an ungated legacy runner')
            require(all(sh['state'] == 'pending' for sh in u['shards'].values()), 'Claims must be drained first')
            require(all(c['state']=='pending' and not c['attempted'] for sh in u['shards'].values() for c in sh['cells'].values()), 'Only a wholly unattempted seed can move clusters')
            proof(e, 'old_admission_excludes_unit', 'old_jobs_cancelled', 'no_queued_or_running_cells')
            destination = e.get('destination')
            require(destination in s['clusters'] and destination != actor, 'Invalid destination')
            rid, r = resource(s, destination, {**u, 'hardware':None}, e, now)
            u.update(owner=destination, epoch=u['epoch']+1, hardware=r['hardware'], transfer_receipt=e['receipt_sha256'])
            result = dict(owner=destination, epoch=u['epoch'], allocation_id=rid)
        elif action == 'claim':
            require(u['mode'] == 'managed', 'Legacy runner still owns admission')
            sh = u['shards'][req['shard']]
            require(sh['state'] == 'pending', 'Shard already claimed/started/done')
            rid, r = resource(s, actor, u, e, now)
            active = active_shards(s, actor)
            require(not any(x.get('worker')==req['worker'] for x in active), 'Worker already has an active shard')
            require(sum(x.get('allocation_id')==rid for x in active) < r['slots']*r['workers_per_gpu'], 'Allocation slots are occupied')
            active_allocations = {sh['allocation_id'] for sh in active}
            require(sum(x['slots'] for key, x in s['resources'].get(actor,{}).items()
                        if key in active_allocations or -30 <= (now-parse_time(x['updated_at'])).total_seconds() <= 300)
                    <= s['clusters'][actor]['max_gpu_slots'], 'Registered GPU allocations exceed cap')
            u['hardware'] = r['hardware']
            token = str(uuid.uuid4())
            sh.update(state='claimed', cluster=actor, worker=req['worker'], allocation_id=rid,
                      epoch=u['epoch'], token=token, heartbeat=stamp(now))
            result = dict(token=token, epoch=u['epoch'], allocation_id=rid)
        elif action in {'heartbeat', 'start-cell', 'finish-cell', 'finish-shard', 'recover'}:
            sh = shard_token(u, req)
            require(sh['state'] in {'claimed','running'}, 'Shard is not active')
            if action == 'heartbeat':
                sh['heartbeat'] = stamp(now)
            elif action == 'start-cell':
                require(-30 <= (now-parse_time(sh['heartbeat'])).total_seconds() <= 180, 'Heartbeat stale; no new cell may start')
                resource(s, actor, u, {'allocation_id':sh['allocation_id']}, now)
                require(not any(c['state']=='running' for c in sh['cells'].values()), 'Worker still has an in-flight cell')
                cell = sh['cells'][req['cell']]
                require(cell['state']=='pending' and not cell['attempted'], 'Cell has already been attempted')
                attempt = str(uuid.uuid4())
                cell.update(state='running', attempted=True, attempt=attempt, started_at=stamp(now))
                sh.update(state='running', heartbeat=stamp(now))
                result = dict(attempt=attempt, token=sh['token'])
            elif action == 'finish-cell':
                cell = sh['cells'][req['cell']]
                require(cell['state']=='running' and cell.get('attempt')==req.get('attempt'), 'Wrong or already finished attempt')
                require(e.get('outcome') in {'complete','failed'}, 'Technical state must be complete or failed')
                require(is_sha256(e.get('result_sha256')), 'Result receipt hash required')
                cell.update(state=e['outcome'], result_sha256=e['result_sha256'], finished_at=stamp(now))
                sh['heartbeat'] = stamp(now)
            elif action == 'finish-shard':
                require(all(c['state'] in TERMINAL for c in sh['cells'].values()), 'Unfinished cells remain')
                sh.update(state='done', finished_at=stamp(now))
            elif action == 'recover':
                proof(e, 'old_worker_stopped', 'old_jobs_cancelled', 'no_queued_or_running_cells')
                for cell in sh['cells'].values():
                    if cell['state']=='running':
                        cell.update(state='interrupted', stop_receipt=e['receipt_sha256'])
                sh.update(state='done' if all(c['state'] in TERMINAL for c in sh['cells'].values()) else 'pending',
                          token=None, worker=None, stop_receipt=e['receipt_sha256'])
        else:
            raise ValueError('Unknown action')
    s['revision'] += 1
    event = dict(id=req.get('operation_id') or str(uuid.uuid4()), revision=s['revision'],
                 action=action, actor=actor, unit=req.get('unit'), shard=req.get('shard'), at=stamp(now),
                 evidence_sha256=digest(e))
    s['events'].append(event)
    return s, {**result, 'event_id':event['id'], 'revision':s['revision']}


def git(path, *args):
    return subprocess.check_output(['git','-C',str(path),*args], text=True, stderr=subprocess.PIPE).strip()


class GitStore:
    def __init__(self, directory, remote, branch):
        require(not branch.startswith('-') and '..' not in branch, 'Invalid branch')
        self.directory = Path(directory)
        marker = self.directory / '.coord-owned'
        if not marker.exists():
            require(not self.directory.exists() or not any(self.directory.iterdir()), 'Use a new dedicated checkout')
            self.directory.mkdir(parents=True, exist_ok=True)
            git(self.directory,'init','-q')
            git(self.directory,'remote','add','origin',remote)
            git(self.directory,'config','user.name','cluster-coordinator')
            git(self.directory,'config','user.email','cluster-coordinator@localhost')
            marker.write_text(json.dumps({'remote':remote,'branch':branch}))
        require(json.loads(marker.read_text())=={'remote':remote,'branch':branch}, 'Checkout remote/branch mismatch')
        self.branch = branch

    def snapshot(self):
        git(self.directory,'fetch','-q','--depth=1','origin','refs/heads/'+self.branch)
        # This checkout is exclusively owned by this utility; no research files are edited here.
        git(self.directory,'checkout','-q','--detach','--force','FETCH_HEAD')
        self.base = git(self.directory,'rev-parse','HEAD')
        return json.loads((self.directory/'registry.json').read_text())

    def commit(self, state, result):
        require(git(self.directory,'rev-parse','HEAD')==self.base, 'Local checkout changed during transaction')
        (self.directory/'registry.json').write_text(json.dumps(state,indent=2)+'\n')
        git(self.directory,'add','registry.json')
        git(self.directory,'commit','-q','-m','coord: '+result['event_id'])
        # No force, merge, rebase or blind retry. Competing updates reject this push.
        git(self.directory,'push','origin','HEAD:refs/heads/'+self.branch)
        tip = git(self.directory,'rev-parse','HEAD')
        observed = self.snapshot()
        require(any(e['id']==result['event_id'] for e in observed['events']), 'Commit could not be confirmed; do not execute')
        return tip


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote',required=True)
    p.add_argument('--branch',default='coordination/gpu02-eval-assist-20261001')
    p.add_argument('--checkout',required=True,help='New directory dedicated to this worker; not an existing repo')
    p.add_argument('--request',help='JSON transition request; omit for read-only status')
    a = p.parse_args()
    store = GitStore(a.checkout,a.remote,a.branch)
    with (store.directory/'.coord-lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state = store.snapshot()
        if not a.request:
            print(json.dumps(state,indent=2));return
        request = json.loads(Path(a.request).read_text())
        updated,result = apply(state,request)
        result['commit'] = store.commit(updated,result)
        print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
