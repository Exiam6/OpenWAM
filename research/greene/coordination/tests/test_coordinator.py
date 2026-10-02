"""Scheduling invariants and real non-fast-forward races; no GPU/network required."""
import copy
import datetime as dt
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('coord', Path(__file__).parents[1] / 'coordinator.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
NOW = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)
HASH = 'a' * 64


def state():
    cell = {'state': 'pending', 'attempted': False}
    unit = {'owner': 'gpu02', 'mode': 'managed', 'epoch': 1, 'hardware': None,
            'manifest_sha256': HASH, 'shards': {
                'task-a': {'state': 'pending', 'cells': {'cell-a': copy.deepcopy(cell), 'cell-b': copy.deepcopy(cell)}},
                'task-b': {'state': 'pending', 'cells': {'cell-c': copy.deepcopy(cell)}}}}
    return {'revision': 0, 'events': [], 'new_work_deadline': '2026-10-03T00:00:00+00:00',
            'clusters': {'gpu02': {'max_gpu_slots': 1}, 'torch': {'max_gpu_slots': 2}},
            'resources': {}, 'units': {'seed-a': unit, 'seed-b': copy.deepcopy(unit)}}


def allocation(actor='gpu02', allocation_id='gpu-5', **kwargs):
    return {'action': 'advertise', 'actor': actor, 'evidence': {
        'allocation_id': allocation_id, 'hardware': 'L40S', 'slots': 1, 'verified_gpu_limit': 2,
        'receipt_sha256': HASH, 'allocated': True, 'qos_verified': True, 'workers_per_gpu': 1,
        'assets_verified': True, 'smoke_passed': True, 'runner_admission_verified': True,
        'verified_manifests': [HASH], **kwargs}}


def claim(**kwargs):
    return {'action': 'claim', 'actor': 'gpu02', 'unit': 'seed-a', 'shard': 'task-a',
            'worker': 'worker-1', 'evidence': {'allocation_id': 'gpu-5'}, **kwargs}


class Invariants(unittest.TestCase):
    def setUp(self):
        self.s, _ = c.apply(state(), allocation(), NOW)

    def step(self, req, now=NOW):
        self.s, result = c.apply(self.s, req, now)
        return result

    def start(self):
        r = self.step(claim())
        self.owner = {k: v for k, v in claim().items() if k != 'evidence'}
        self.owner['token'] = r['token']
        return self.step({**self.owner, 'action': 'start-cell', 'cell': 'cell-a'})

    def test_legacy_and_wrong_owner_cannot_claim(self):
        self.s['units']['seed-a']['mode'] = 'legacy'
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            self.step(claim())
        with self.assertRaisesRegex(ValueError, 'another cluster'):
            self.step(claim(actor='torch'))

    def test_actual_assets_manifest_and_smoke_required(self):
        for missing in ('assets_verified', 'smoke_passed', 'runner_admission_verified'):
            with self.subTest(missing=missing):
                self.step(allocation(**{missing: False}))
                with self.assertRaises(ValueError):
                    self.step(claim())
                self.step(allocation())
        self.step(allocation(verified_manifests=[]))
        with self.assertRaisesRegex(ValueError, 'exact unit'):
            self.step(claim())

    def test_attempt_is_once_even_when_failed(self):
        result = self.start()
        with self.assertRaisesRegex(ValueError, 'in-flight'):
            self.step({**self.owner, 'action': 'start-cell', 'cell': 'cell-b'})
        self.step({**self.owner, 'action': 'finish-cell', 'cell': 'cell-a', 'attempt': result['attempt'],
                   'evidence': {'outcome': 'failed', 'result_sha256': HASH}})
        with self.assertRaisesRegex(ValueError, 'already been attempted'):
            self.step({**self.owner, 'action': 'start-cell', 'cell': 'cell-a'})
        with self.assertRaisesRegex(ValueError, 'Unfinished'):
            self.step({**self.owner, 'action': 'finish-shard'})

    def test_stale_does_not_release_ownership(self):
        self.start()
        later = NOW + dt.timedelta(minutes=6)
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            self.step(claim(worker='new-worker'), later)
        with self.assertRaisesRegex(ValueError, 'Missing affirmative'):
            self.step({**self.owner, 'action': 'recover', 'evidence': {'receipt_sha256': HASH}}, later)
        self.assertEqual(self.s['units']['seed-a']['shards']['task-a']['state'], 'running')

    def test_stale_heartbeat_and_resource_block_new_work(self):
        result = self.start()
        self.step({**self.owner, 'action': 'finish-cell', 'cell': 'cell-a', 'attempt': result['attempt'],
                   'evidence': {'outcome': 'complete', 'result_sha256': HASH}})
        with self.assertRaisesRegex(ValueError, 'Heartbeat stale'):
            self.step({**self.owner, 'action': 'start-cell', 'cell': 'cell-b'}, NOW + dt.timedelta(seconds=181))
        later = NOW + dt.timedelta(minutes=6)
        self.step({**self.owner, 'action': 'heartbeat'}, later)
        with self.assertRaisesRegex(ValueError, 'Resource report stale'):
            self.step({**self.owner, 'action': 'start-cell', 'cell': 'cell-b'}, later)

    def test_recovery_fences_old_token_and_preserves_attempt(self):
        self.start()
        self.step({**self.owner, 'action': 'recover', 'evidence': {
            'receipt_sha256': HASH, 'old_worker_stopped': True, 'old_jobs_cancelled': True,
            'no_queued_or_running_cells': True}})
        self.assertEqual(self.s['units']['seed-a']['shards']['task-a']['cells']['cell-a']['state'], 'interrupted')
        new = self.step(claim())
        self.assertNotEqual(new['token'], self.owner['token'])
        with self.assertRaisesRegex(ValueError, 'fencing token'):
            self.step({**self.owner, 'action': 'heartbeat'})
        with self.assertRaisesRegex(ValueError, 'already been attempted'):
            self.step({**self.owner, 'token': new['token'], 'action': 'start-cell', 'cell': 'cell-a'})

    def test_capacity_and_hardware_cannot_be_silently_expanded(self):
        self.start()
        with self.assertRaisesRegex(ValueError, 'slots are occupied'):
            self.step(claim(shard='task-b', worker='worker-2'))
        with self.assertRaisesRegex(ValueError, 'active allocation hardware'):
            self.step(allocation(hardware='H200'))
        with self.assertRaisesRegex(ValueError, 'Missing affirmative'):
            self.step(allocation(workers_per_gpu=2))
        self.step(allocation(workers_per_gpu=2, parallel_smoke_passed=True, parity_passed=True,
                             no_oom=True, throughput_gain=1.2, peak_memory_fraction=.7))
        self.step(claim(shard='task-b', worker='worker-2'))
        with self.assertRaisesRegex(ValueError, 'shrink'):
            self.step(allocation())

    def test_stale_active_allocation_still_counts_toward_cap(self):
        self.start()
        later = NOW + dt.timedelta(minutes=10)
        self.step(allocation(allocation_id='gpu-6'), later)
        with self.assertRaisesRegex(ValueError, 'allocations exceed cap'):
            self.step(claim(unit='seed-b', worker='worker-2', evidence={'allocation_id': 'gpu-6'}), later)

    def transfer_request(self):
        return {'action': 'transfer', 'actor': 'gpu02', 'unit': 'seed-a', 'evidence': {
            'destination': 'torch', 'allocation_id': 'slurm-1', 'receipt_sha256': HASH,
            'old_admission_excludes_unit': True, 'old_jobs_cancelled': True,
            'no_queued_or_running_cells': True}}

    def test_transfer_requires_ready_destination_and_unattempted_seed(self):
        req = self.transfer_request()
        with self.assertRaisesRegex(ValueError, 'not registered'):
            self.step(req)
        self.step(allocation('torch', 'slurm-1', hardware='H200'))
        self.start()
        with self.assertRaisesRegex(ValueError, 'drained'):
            self.step(req)
        self.s['units']['seed-a'] = copy.deepcopy(self.s['units']['seed-b'])
        self.s['units']['seed-a']['shards']['task-a']['cells']['cell-a'].update(state='failed', attempted=True)
        with self.assertRaisesRegex(ValueError, 'wholly unattempted'):
            self.step(req)

    def test_transfer_changes_owner_epoch_and_pins_hardware(self):
        self.step(allocation('torch', 'slurm-1', hardware='H200'))
        result = self.step(self.transfer_request())
        self.assertEqual(result['epoch'], 2)
        with self.assertRaisesRegex(ValueError, 'another cluster'):
            self.step(claim())
        self.step(claim(actor='torch', evidence={'allocation_id': 'slurm-1'}))
        self.step(allocation('torch', 'slurm-2'))
        with self.assertRaisesRegex(ValueError, 'Hardware class'):
            self.step(claim(actor='torch', shard='task-b', worker='worker-2', evidence={'allocation_id': 'slurm-2'}))

    def test_adoption_requires_complete_stopped_inventory(self):
        u = self.s['units']['seed-a']
        u['mode'] = 'legacy'
        u['shards']['task-a']['cells']['cell-a'].update(state='running', attempted=True)
        evidence = dict(receipt_sha256=HASH, legacy_admission_disabled=True,
                        no_queued_or_running_cells=True, gated_runner_verified=True, inventory_verified=True,
                        cell_states={'cell-a':'pending','cell-b':'pending','cell-c':'pending'})
        req = {'action':'adopt','actor':'gpu02','unit':'seed-a','evidence':evidence}
        with self.assertRaisesRegex(ValueError, 'cannot become pending'):
            self.step(req)
        evidence['cell_states']['cell-a'] = 'interrupted'
        self.step(req)
        self.assertEqual(self.s['units']['seed-a']['mode'], 'managed')

    def test_deadline_and_duplicate_operation_id(self):
        req = {**allocation(), 'operation_id': 'once'}
        self.step(req)
        with self.assertRaisesRegex(ValueError, 'already recorded'):
            self.step(req)
        with self.assertRaisesRegex(ValueError, 'deadline passed'):
            self.step(claim(), NOW + dt.timedelta(days=3))


class GitRace(unittest.TestCase):
    def test_two_stale_claims_cannot_both_push(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root/'remote.git'
            subprocess.check_call(['git', 'init', '-q', '--bare', str(remote)])
            seed = root/'seed'
            seed.mkdir()
            c.git(seed, 'init', '-q')
            c.git(seed, 'config', 'user.name', 'test')
            c.git(seed, 'config', 'user.email', 'test@localhost')
            initial, _ = c.apply(state(), allocation(), NOW)
            (seed/'registry.json').write_text(json.dumps(initial))
            c.git(seed, 'add', '.')
            c.git(seed, 'commit', '-qm', 'initial')
            c.git(seed, 'push', str(remote), 'HEAD:refs/heads/coord-test')
            a, b = [c.GitStore(root/name, str(remote), 'coord-test') for name in ('a','b')]
            sa, sb = a.snapshot(), b.snapshot()
            sa, ra = c.apply(sa, claim(), NOW)
            sb, rb = c.apply(sb, claim(worker='worker-2'), NOW)
            a.commit(sa, ra)
            with self.assertRaises(subprocess.CalledProcessError):
                b.commit(sb, rb)
            winner = b.snapshot()
            self.assertEqual(winner['units']['seed-a']['shards']['task-a']['worker'], 'worker-1')
            with self.assertRaisesRegex(ValueError, 'already claimed'):
                c.apply(winner, claim(worker='worker-2'), NOW)


if __name__ == '__main__':
    unittest.main()
