import json
from pathlib import Path
import tempfile
import unittest
from corrected_evidence import prior_cost,release_path
from build_cache import sha
import report_complete


class CorrectedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        def write(name,value):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
        self.write=write
        protocol=write('corrected-protocol-v1/protocol.json',{'status':'registered_corrected_protocol'})
        pre=write('dedicated-preflight-run.json',dict(status='failed',passed=False,occupied_seconds=200,stages=[dict(name='cache',execution=dict(occupied_seconds=100))]))
        evidence={str(pre):sha(pre)}
        for name in ['deployment-diagnostic-0807','encoder-precision-diagnostic-0812','fixed-policy-diagnostic-0825','corrected-tiny-validation-0827']:
            p=write(name+'/execution.json',dict(status='complete',original_evidence_unchanged=True,execution=dict(complete=True,exit_code=0,occupied_seconds=1)))
            evidence[str(p)]=sha(p)
        self.release=dict(original_preflight_status='failed',corrected_protocol_sha256=sha(protocol),evidence_sha256=evidence,prior_occupied_gpu_seconds=204)

    def test_failed_attempt_and_all_diagnostics_charged(self):
        self.assertEqual(prior_cost(self.root,self.release),204)
        self.assertEqual(report_complete.selected_preflight(self.root,self.release),self.root/'dedicated-preflight-run.json')

    def test_underreported_cost_rejected(self):
        with self.assertRaises(AssertionError):prior_cost(self.root,dict(self.release,prior_occupied_gpu_seconds=203))

    def test_failed_preflight_cannot_be_relabelled(self):
        self.write('dedicated-preflight-run.json',dict(status='complete',passed=True,occupied_seconds=200))
        with self.assertRaises(AssertionError):prior_cost(self.root,self.release)

    def test_corrected_release_selected_explicitly(self):
        self.write('full-training-release.json',{'passed':True})
        self.assertEqual(release_path(self.root).name,'full-training-release.json')
        self.write('corrected-full-training-release-v1.json',self.release)
        self.assertEqual(release_path(self.root).name,'corrected-full-training-release-v1.json')


if __name__=='__main__':unittest.main()


class InitializationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.parent=Path(self.temp.name);self.root=self.parent/'study';self.root.mkdir()
        self.source=self.parent/'rae-factorial-20261010/initialization/common-seed42.safetensors';self.source.parent.mkdir(parents=True);self.source.write_bytes(b'frozen shared initialization')
        (self.root/'gpu-registration.json').write_text(json.dumps({'asset_files':{str(self.source):sha(self.source)}}))
        self.path=self.root/'runs/vae48_native-seed42/initialization.json';self.path.parent.mkdir(parents=True)
        self.record={'seed':42,'trained_official_policy_weights_loaded':False,'cuda_initialized':False,'matched_trainable_tensors':676,'common_initialization':str(self.source),'dimension_specific_exclusions':[dict(name=x,raw_shape=[768],vae_shape=[48]) for x in ['video_backbone.dit.patch_embedding.weight','video_backbone.dit.head.head.weight','video_backbone.dit.head.head.bias']]}
        self.path.write_text(json.dumps(self.record))

    def test_fresh_correct_seed_is_archivable(self):
        from corrected_evidence import verify_initialization
        self.assertEqual(verify_initialization(self.root,42),self.path)

    def test_wrong_seed_or_trained_weights_rejected(self):
        from corrected_evidence import verify_initialization
        for field,value in [('seed',43),('trained_official_policy_weights_loaded',True),('matched_trainable_tensors',675)]:
            self.path.write_text(json.dumps(dict(self.record,**{field:value})))
            with self.assertRaises(AssertionError):verify_initialization(self.root,42)

    def test_changed_shared_initialization_rejected(self):
        from corrected_evidence import verify_initialization
        self.source.write_bytes(b'changed')
        with self.assertRaises(AssertionError):verify_initialization(self.root,42)
