import json
from pathlib import Path
import tempfile
import unittest

from build_cache import sha
from report_complete import pilot_dispatch_evidence


class PilotDispatchArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.put('pilot-capacity-code-v1/pilot_queue.py','frozen source')
        protocol=self.put('corrected-protocol-v1/protocol.json',{})
        runtime=self.put('corrected-engine-v1-manifest.json',{})
        self.queue=dict(pid=123,status='complete',full_evaluation_automatically_released=False,
                        active_bounded_parents={},seeds={})
        audit=dict(passed=True,policy_success_claim=False,rows=[])
        initializations={}
        for seed in [42,43,44]:
            pilot=self.put(f'evaluation/seed{seed}/pilot/result.json',
                           dict(seed=seed,complete=True,episodes=15))
            claim=self.put(f'phase-claims/pilot_evaluation-seed{seed}.json',
                           dict(seed=seed,complete=True,exit_code=0,result_sha256=sha(pilot)))
            self.queue['seeds'][str(seed)]=dict(status='complete',returncode=0,
                result=str(pilot),result_sha256=sha(pilot),parent_claim_sha256=sha(claim))
            self.put(f'pilot-capacity-queue-v1/seed{seed}.log','completed')
            self.put(f'batches/pilot_evaluation-{seed}/result.json',dict(complete=True))
            self.put(f'batches/pilot_evaluation-{seed}/seed{seed}/run.log','bounded parent log')
            run=f'runs/vae48_native-seed{seed}'
            trained=self.put(run+'/result.json',dict(checkpoint_sha256=f'checkpoint{seed}'))
            init=self.put(run+'/initialization.json',dict(seed=seed));initializations[str(seed)]=sha(init)
            metrics=self.put(run+'/metrics.jsonl','fixed endpoint metrics')
            audit['rows'].append(dict(seed=seed,passed=True,steps=20000,
                checkpoint_hash_independently_recomputed=True,checkpoint_sha256=f'checkpoint{seed}',
                evidence=dict(result_sha256=sha(trained)),file_sha256={'metrics.jsonl':sha(metrics)}))
        self.put('pilot-capacity-queue-v1.log','CPU queue log')
        self.put('full-training-independent-audit-v1.json',audit)
        self.put('full-training-independent-audit-observer-fix-v1/record.json',dict(observer_only=True))
        self.put('full-training-independent-audit-observer-fix-v1/original-observer.py','original observer')
        self.put('full-training-independent-audit-observer-fix-v1/corrected-observer.py','corrected observer')
        plan=dict(status='registered_resource_only',phase='pilot_evaluation',seeds=[42,43,44],
                  full_evaluation_automatically_released=False,files={str(self.source):sha(self.source)},
                  protocol_sha256=sha(protocol),runtime_manifest_sha256=sha(runtime),
                  initialization_sha256=initializations)
        plan_path=self.put('pilot-capacity-queue-v1-plan.json',plan)
        self.queue['plan_sha256']=sha(plan_path)
        self.put('pilot-capacity-queue-v1-launch.json',dict(pid=123,plan_sha256=sha(plan_path)))
        self.persist_queue()

    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value,str) else value)
        return path

    def persist_queue(self):self.put('pilot-capacity-queue-v1/result.json',self.queue)

    def test_complete_queue_and_observer_correction_history_are_archived(self):
        expected={p for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(set(pilot_dispatch_evidence(self.root)),expected)

    def test_pending_queue_cannot_produce_final_evidence(self):
        self.queue['status']='pilots_running';self.persist_queue()
        with self.assertRaises(AssertionError):pilot_dispatch_evidence(self.root)

    def test_automatic_full_release_is_rejected(self):
        self.queue['full_evaluation_automatically_released']=True;self.persist_queue()
        with self.assertRaises(AssertionError):pilot_dispatch_evidence(self.root)

    def test_modified_registered_source_is_rejected(self):
        self.source.write_text('changed')
        with self.assertRaises(AssertionError):pilot_dispatch_evidence(self.root)

    def test_modified_pilot_or_parent_is_rejected(self):
        self.put('evaluation/seed42/pilot/result.json',{})
        with self.assertRaises(AssertionError):pilot_dispatch_evidence(self.root)

    def test_missing_launch_log_is_rejected(self):
        (self.root/'pilot-capacity-queue-v1/seed43.log').unlink()
        with self.assertRaises(AssertionError):pilot_dispatch_evidence(self.root)

    def test_changed_training_metrics_are_rejected(self):
        self.put('runs/vae48_native-seed44/metrics.jsonl','changed')
        with self.assertRaises(AssertionError):pilot_dispatch_evidence(self.root)

    def test_completed_media_process_is_retained(self):
        source=self.put('full-evidence-code-v8/full_media.py','original media source')
        self.put('full-media-process-v8.json',dict(pid=456,command=['python',str(source)],source_sha256=sha(source)))
        self.put('full-media/queue.json',dict(pid=456,status='complete',completed=360))
        self.put('full-media-v8.log','media log')
        paths=set(pilot_dispatch_evidence(self.root))
        self.assertIn(source,paths)
        self.assertIn(self.root/'full-media-process-v8.json',paths)
        self.put('full-media/queue.json',dict(pid=456,status='encoding',completed=359))
        with self.assertRaises(AssertionError):pilot_dispatch_evidence(self.root)

    def test_report_snapshot_and_tests_are_hash_checked_and_archived(self):
        source=self.put('full-evidence-code-v9/report_complete.py','report source')
        tests=self.put('full-evidence-v9-tests.json',dict(passed=True,returncode=0))
        self.put('full-evidence-code-v9-manifest.json',dict(archive_provenance_only=True,
            files={str(source):sha(source)},tests_record=str(tests),tests_record_sha256=sha(tests)))
        paths=set(pilot_dispatch_evidence(self.root))
        self.assertTrue({source,tests,self.root/'full-evidence-code-v9-manifest.json'}<=paths)
        source.write_text('changed source')
        with self.assertRaises(AssertionError):pilot_dispatch_evidence(self.root)


if __name__=='__main__':unittest.main()
