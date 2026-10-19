import json
from pathlib import Path
import tempfile
import unittest
from resumption_evidence import evaluation_cap,evidence
from build_cache import sha


class ResumptionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.auth=self.root/'resumption-20261019-v1/authorization.json'

    def put(self,**changes):
        self.auth.parent.mkdir(exist_ok=True)
        data=dict(status='authorized_resource_resumption',evaluation_limits_seconds={'42':10920,'43':10920,'44':10800},study_limit_seconds=73800)
        data.update(changes);self.auth.write_text(json.dumps(data))

    def test_original_caps_without_amendment(self):
        self.assertEqual(evaluation_cap(self.root,42),10800)
        self.assertEqual(evidence(self.root),[])

    def test_only_registered_caps_accepted(self):
        self.put()
        self.assertEqual(evaluation_cap(self.root,42),10920)
        self.assertEqual(evaluation_cap(self.root,44),10800)
        self.put(evaluation_limits_seconds={'42':10921,'43':10920,'44':10800})
        with self.assertRaises(AssertionError):evaluation_cap(self.root,42)

    def test_proposal_is_not_authorization(self):
        self.put(status='proposal_only_pending_explicit_resource_authorization')
        with self.assertRaises(AssertionError):evaluation_cap(self.root,42)

    def test_global_cap_cannot_increase(self):
        self.put(study_limit_seconds=73801)
        with self.assertRaises(AssertionError):evaluation_cap(self.root,42)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        def put(name,data):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(data));return path
        self.put=put
        auth=put('resumption-20261019-v1/authorization.json',dict(status='authorized_resource_resumption',seeds=[42,43],gpus=[1,2],no_retraining=True,no_automatic_retry=True,evidence_sha256={},evaluation_limits_seconds={'42':10920,'43':10920,'44':10800},study_limit_seconds=73800))
        self.auth_sha=sha(auth)
        for name in ['resumption-code-v1-manifest.json','resumption-evidence-code-v1-manifest.json']:
            put(name,dict(authorization_sha256=self.auth_sha,files={}))
        for name in ['frozen-tests.json','evidence-tests.json']:
            put('resumption-20261019-v1/'+name,dict(passed=True,tests=1))
        for name in ['batch-process.json','media-process.json','batch.log','media.log']:
            put('resumption-20261019-v1/'+name,{})
        put('batches/full_evaluation-42-43/result.json',dict(complete=True,status='complete',seeds=[42,43],gpus=[1,2],children=[dict(complete=True,exit_code=0)]*2))
        for seed in [42,43]:
            pilot=put(f'phase-claims/pilot_evaluation-seed{seed}.json',dict(occupied_seconds=2500))
            put(f'evaluation-release-seed{seed}.json',dict(resource_authorization_sha256=self.auth_sha,evaluation_limit_seconds=10920,parent_claim_sha256=sha(pilot),remaining_seconds=8420,cost=dict(projected_remaining_seconds=8300)))
            for name in ['execution.json','run.log']:put(f'batches/full_evaluation-42-43/seed{seed}/{name}',{})
        self.old=put('full-media/queue.json',dict(status='failed',completed=180,active={}))
        put('full-media-v8.log',{})
        inventory=put('resumption-preparation-20261019-v1/verified-media-reuse-inventory.json',dict(verified_pairs=180,missing_pairs=180,source_sha256={str(self.old):sha(self.old)}))
        self.queue=put('full-media-continuation-v1/queue.json',dict(status='complete',completed=360,reused=180,active={},authorization_sha256=self.auth_sha,inventory_sha256=sha(inventory)))
        put('full-media/manifest.json',dict(complete=True,paired_videos=360,continuation_authorization_sha256=self.auth_sha,original_queue_sha256=sha(self.old),reuse_inventory_sha256=sha(inventory)))

    def test_complete_continuation_archives_original_failure(self):
        paths=evidence(self.root)
        self.assertIn(self.old,paths);self.assertIn(self.queue,paths)
        self.assertEqual(json.loads(self.old.read_text())['status'],'failed')

    def test_original_failure_cannot_be_relabelled(self):
        data=json.loads(self.old.read_text());data['status']='complete';self.old.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):evidence(self.root)

    def test_unfinished_continuation_rejected(self):
        data=json.loads(self.queue.read_text());data['status']='encoding';self.queue.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):evidence(self.root)

    def test_release_must_reference_this_authorization(self):
        path=self.root/'evaluation-release-seed42.json';data=json.loads(path.read_text());data['resource_authorization_sha256']='wrong';path.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):evidence(self.root)


if __name__=='__main__':unittest.main()
