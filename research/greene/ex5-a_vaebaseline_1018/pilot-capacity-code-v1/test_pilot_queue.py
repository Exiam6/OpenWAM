import datetime as dt
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pilot_queue as q


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'runs/vae48_native-seed42'
        self.run.mkdir(parents=True)
        self.checkpoint = self.run / 'checkpoint_step_20000.safetensors'
        self.checkpoint.write_bytes(b'checkpoint')
        self.init = self.run / 'initialization.json'
        write(self.init, {'seed': 42})
        self.plan = dict(protocol_sha256='protocol', runtime_manifest_sha256='runtime',
                         initialization_sha256={'42': q.sha(self.init)})
        self.result_path = self.run / 'result.json'
        self.result = dict(seed=42, stage='full', complete=True, steps=20000,
            strict_reload=True, checkpoint=str(self.checkpoint), checkpoint_sha256=q.sha(self.checkpoint),
            corrected_protocol_sha256='protocol', runtime_manifest_sha256='runtime')
        self.claim_path = self.root / 'phase-claims/training-seed42.json'
        self.persist()

    def persist(self):
        write(self.result_path, self.result)
        self.claim = dict(complete=True, exit_code=0, seed=42,
            result_path=str(self.result_path), result_sha256=q.sha(self.result_path))
        write(self.claim_path, self.claim)

    def test_completed_parent_and_fixed_endpoint_are_ready(self):
        self.assertEqual(q.training_evidence(self.root, 42, self.plan)['checkpoint_sha256'],
                         self.result['checkpoint_sha256'])

    def test_missing_parent_claim_waits(self):
        self.claim_path.unlink()
        self.assertIsNone(q.training_evidence(self.root, 42, self.plan))

    def test_successful_child_does_not_bypass_parent_cleanup(self):
        write(self.claim_path, dict(status='running', seed=42, pid=123))
        self.assertIsNone(q.training_evidence(self.root, 42, self.plan))

    def test_failed_parent_is_not_retried(self):
        self.claim.update(complete=False, exit_code=1)
        write(self.claim_path, self.claim)
        before = self.claim_path.read_bytes()
        with self.assertRaises(AssertionError): q.training_evidence(self.root, 42, self.plan)
        self.assertEqual(before, self.claim_path.read_bytes())

    def test_tampered_result_is_rejected(self):
        self.result_path.write_text('{}')
        with self.assertRaises(AssertionError): q.training_evidence(self.root, 42, self.plan)

    def test_nonfinal_checkpoint_is_rejected(self):
        self.result['steps'] = 19975
        self.persist()
        with self.assertRaises(AssertionError): q.training_evidence(self.root, 42, self.plan)

    def test_strict_reload_is_required(self):
        self.result['strict_reload'] = False
        self.persist()
        with self.assertRaises(AssertionError): q.training_evidence(self.root, 42, self.plan)

    def test_checkpoint_outside_run_is_rejected(self):
        self.result['checkpoint'] = str(self.root / 'another.safetensors')
        self.persist()
        with self.assertRaises(AssertionError): q.training_evidence(self.root, 42, self.plan)

    def test_initialization_mutation_is_rejected(self):
        self.init.write_text('{}')
        with self.assertRaises(AssertionError): q.training_evidence(self.root, 42, self.plan)

    def test_wrong_protocol_is_rejected(self):
        self.result['corrected_protocol_sha256'] = 'old-protocol'
        self.persist()
        with self.assertRaises(AssertionError): q.training_evidence(self.root, 42, self.plan)

    def test_prior_grouped_batch_blocks_duplicate_seed(self):
        p = self.root / 'batches/pilot_evaluation-42-43-44'
        p.mkdir(parents=True)
        self.assertIn(str(p), q.prior_attempts(self.root, 42))

    def test_existing_partial_output_blocks_attempt(self):
        p = self.root / 'evaluation/seed42/pilot'
        p.mkdir(parents=True)
        self.assertIn(str(p), q.prior_attempts(self.root, 42))

    def test_plan_mutation_is_rejected_before_loading(self):
        path = self.root / 'pilot-capacity-queue-v1-plan.json'
        write(path, {'state': 'before'})
        digest = q.sha(path)
        write(path, {'state': 'after'})
        with patch.object(q, 'ROOT', self.root), self.assertRaisesRegex(AssertionError, 'changed'):
            q.load_plan(digest)

    def test_gpu_lock_probe_respects_existing_owner(self):
        with (self.root / 'gpu-3.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertFalse(q.lock_available(self.root, 3))
        self.assertTrue(q.lock_available(self.root, 3))


class PilotCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result_path = self.root / 'evaluation/seed42/pilot/result.json'
        self.result = dict(complete=True, status='complete', seed=42, stage='pilot',
                           episodes=15, groups=[dict(episodes=5) for _ in range(3)])
        write(self.root / 'batches/pilot_evaluation-42/result.json',
              dict(status='complete', complete=True, seeds=[42]))
        self.persist()

    def persist(self):
        write(self.result_path, self.result)
        write(self.root / 'phase-claims/pilot_evaluation-seed42.json',
              dict(complete=True, exit_code=0, seed=42, occupied_seconds=1000,
                   result_path=str(self.result_path), result_sha256=q.sha(self.result_path)))

    def test_complete_fifteen_episode_pilot_is_accepted(self):
        self.assertEqual(q.verify_pilot_completion(self.root, 42, 0)['occupied_seconds'], 1000)

    def test_parent_failure_cannot_be_masked_by_successful_result(self):
        with self.assertRaises(AssertionError): q.verify_pilot_completion(self.root, 42, 1)

    def test_incomplete_task_is_rejected(self):
        self.result['groups'][1]['episodes'] = 4
        self.persist()
        with self.assertRaises(AssertionError): q.verify_pilot_completion(self.root, 42, 0)

    def test_changed_result_is_rejected(self):
        self.result_path.write_text('{}')
        with self.assertRaises(AssertionError): q.verify_pilot_completion(self.root, 42, 0)


class CapacityTests(unittest.TestCase):
    def test_complete_remaining_window_is_strict(self):
        deadline = '2026-10-18T17:00:00+00:00'
        edge = dt.datetime.fromisoformat(deadline) - dt.timedelta(seconds=60 + q.REQUIRED_SECONDS)
        self.assertFalse(q.window_open(edge, deadline))
        self.assertTrue(q.window_open(edge - dt.timedelta(seconds=1), deadline))

    def test_reserved_and_locked_gpus_are_not_reused(self):
        samples = {g: dict(idle=True) for g in [0, 1, 2, 3, 4]}
        self.assertEqual(q.choose_gpu(samples, {3}, lambda g: g != 1), 2)
        self.assertIsNone(q.choose_gpu(samples, {1, 2, 3, 4}, lambda g: True))

    def test_idle_gpu3_is_preferred(self):
        self.assertEqual(q.choose_gpu({g: dict(idle=True) for g in [1,2,3,4]}, set(), lambda g: True), 3)

    def test_busy_gpu3_is_not_selected(self):
        samples = {g: dict(idle=g != 3) for g in [1,2,3,4]}
        self.assertEqual(q.choose_gpu(samples, set(), lambda g: True), 1)

    def test_command_cannot_release_full_evaluation(self):
        argv = q.command(42, 3)
        self.assertEqual(argv[argv.index('--phase') + 1], 'pilot_evaluation')
        self.assertEqual(argv[argv.index('--seeds') + 1], '42')
        self.assertEqual(argv[argv.index('--gpus') + 1], '3')
        with self.assertRaises(AssertionError): q.command(45, 3)
        with self.assertRaises(AssertionError): q.command(42, 0)


class ObserverFailureTests(unittest.TestCase):
    def test_observer_error_preserves_bounded_children_and_never_relaunches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root / 'pilot-capacity-queue-v1-plan.json', {})
            registration = dict(evaluation_pilot_seconds=3600,
                evaluation_seconds_per_seed=10800, deadline='2030-01-01T17:00:00+00:00', pause_paths=[])
            module = types.SimpleNamespace(load_registration=lambda: registration,
                idle_capacity=lambda p, gs: {g: dict(idle=True) for g in gs})
            children = [Mock(pid=100001), Mock(pid=100002)]
            for child in children: child.poll.return_value = None
            loops = []
            def sleep(_):
                loops.append(1)
                if len(loops) == 2: raise RuntimeError('simulated observer fault')
            with patch.object(q, 'ROOT', root), patch.object(q, 'ENGINE', root), \
                 patch.object(q, 'load_plan', return_value={}), \
                 patch.object(q, 'training_evidence', side_effect=lambda r,s,p: {'verified': True} if s != 44 else None), \
                 patch.dict(sys.modules, {'corrected_protocol': module}), \
                 patch.object(q.signal, 'signal'), \
                 patch.object(q.subprocess, 'Popen', side_effect=children) as spawn, \
                 patch.object(q.time, 'sleep', side_effect=sleep):
                with self.assertRaisesRegex(RuntimeError, 'observer fault'): q.main()
            self.assertEqual(spawn.call_count, 2)
            commands = [call.args[0] for call in spawn.call_args_list]
            self.assertEqual([x[x.index('--gpus') + 1] for x in commands], ['3', '1'])
            self.assertEqual([x[x.index('--seeds') + 1] for x in commands], ['42', '43'])
            for child in children:
                child.terminate.assert_not_called()
                child.kill.assert_not_called()
            state = q.read(root / 'pilot-capacity-queue-v1/result.json')
            self.assertEqual(state['status'], 'observer_failed')
            self.assertEqual(state['active_bounded_parents'], {'42': 100001, '43': 100002})
            self.assertFalse(state['full_evaluation_automatically_released'])

    def test_expired_window_launches_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root / 'pilot-capacity-queue-v1-plan.json', {})
            registration = dict(evaluation_pilot_seconds=3600,
                evaluation_seconds_per_seed=10800, deadline='2000-01-01T17:00:00+00:00', pause_paths=[])
            module = types.SimpleNamespace(load_registration=lambda: registration, idle_capacity=Mock())
            with patch.object(q, 'ROOT', root), patch.object(q, 'load_plan', return_value={}), \
                 patch.dict(sys.modules, {'corrected_protocol': module}), \
                 patch.object(q.signal, 'signal'), patch.object(q.subprocess, 'Popen') as spawn:
                q.main()
            spawn.assert_not_called()
            module.idle_capacity.assert_not_called()
            state = q.read(root / 'pilot-capacity-queue-v1/result.json')
            self.assertEqual(state['status'], 'stopped_incomplete')


if __name__ == '__main__':
    unittest.main()
