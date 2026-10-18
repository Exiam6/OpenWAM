"""Resource provenance must retain registered bytes and reject altered evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import report_complete as report

class ResourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'ex5-a_vaebaseline_1018';self.root.mkdir()
        def put(name,data):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(data));return path
        self.put=put
        registry=put('gpu-registration.json',{})
        self.numeric=put('code/fixture.py',{'numerical':'unchanged'})
        probe=put('resource-code-v2/fixture.py',{'resource':'corrected'})
        manifest=put('resource-registration-v2.json',dict(numerical_stage_files_unchanged=True,actual_used_memory_limit_mib=256,gpu_registration_sha256=report.sha(registry),files={str(probe):report.sha(probe)},original_runtime_files={str(self.numeric):report.sha(self.numeric)}))
        self.old=put('preflight-schedule.json',{'status':'waiting'})
        put('preflight-schedule-v2.json',{'status':'complete'})
        put('resource-scheduler-replacement-v2.json',dict(resource_registration_sha256=report.sha(manifest),registration_sha256=report.sha(registry),scientific_attempt_started=False,admission_full_asset_hash_check_passed=True,old_state_path=str(self.old),old_state_sha256=report.sha(self.old)))

    def test_all_resource_evidence_retained(self):
        self.assertEqual(set(report.resource_evidence(self.root)),{p for p in self.root.rglob('*') if p.is_file()})

    def joint_fixture(self):
        groot=self.root.parent/'ex4-g_calibratedfactorial_1017';groot.mkdir()
        for name in ['registration.json','joint-allocation-process.json','worker-gpu-joint-allocation.json']:(groot/name).write_text('{}')
        old=self.put('preflight-run.json',dict(status='failed',stages=[]))
        log=self.root/'scheduled-preflight-v2.log';log.write_text('capacity expiry')
        expiry=self.put('extra-window-admission-expired.json',dict(scientific_gpu_attempt_started=False,occupied_gpu_seconds=0,
            preflight_state_sha256=report.sha(old),scheduler_state_sha256=report.sha(self.root/'preflight-schedule-v2.json'),log_sha256=report.sha(log)))
        cpu=self.put('joint-resource-code/fixture.py',dict(cpu_only=True))
        self.put('joint-allocation-plan.json',dict(status='registered_resource_only',g_registration_sha256=report.sha(groot/'registration.json'),
            v_registration_sha256=report.sha(self.root/'gpu-registration.json'),files={str(cpu):report.sha(cpu),str(expiry):report.sha(expiry)}))
        self.put('dedicated-preflight-run.json',dict(status='complete'));self.put('joint-allocation.json',{})
        return cpu,old

    def test_both_capacity_histories_and_joint_source_retained(self):
        cpu,old=self.joint_fixture();paths=set(report.resource_evidence(self.root))
        self.assertTrue({cpu,old,self.root/'dedicated-preflight-run.json',self.root/'joint-allocation.json',self.root/'scheduled-preflight-v2.log'}<=paths)
        self.assertIn(self.root.parent/'ex4-g_calibratedfactorial_1017'/'worker-gpu-joint-allocation.json',paths)

    def test_joint_source_and_expired_history_cannot_change(self):
        cpu,old=self.joint_fixture();before=cpu.read_bytes();cpu.write_text('changed')
        with self.assertRaises(AssertionError):report.resource_evidence(self.root)
        cpu.write_bytes(before);old.write_text('changed')
        with self.assertRaises(AssertionError):report.resource_evidence(self.root)

    def test_numerical_file_mutation_rejected(self):
        self.numeric.write_text('changed')
        with self.assertRaises(AssertionError):report.resource_evidence(self.root)

    def test_prior_state_mutation_rejected(self):
        self.old.write_text('{}')
        with self.assertRaises(AssertionError):report.resource_evidence(self.root)

if __name__=='__main__':unittest.main()
