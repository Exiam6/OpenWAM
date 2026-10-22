import copy
from datetime import datetime, timedelta, timezone
import unittest
from audit_outcome import assess, PHASES, PREWARM

class Chain(unittest.TestCase):
    def fixture(self):
        start=datetime(2026,10,21,tzinfo=timezone.utc)
        job=dict(id="registered-job",scientific_identity="registered/attempt1",
                 host_commands={"gpu01":{"argv":["python3","frozen.py"]}})
        samples=[dict(phase=p,time=(start+timedelta(seconds=i*3+j+1)).isoformat(),
                      contexts=[dict(uuid="GPU-assigned")]) for i,p in enumerate(PHASES) for j in range(3)]
        scientific=dict(status="glfw_prefix_passed",scored_episodes=0,samples=samples,
                        prewarm_stdout=PREWARM,monitor_errors=[],glfw_init=1,glfw_vulkan_supported=1,
                        discovered_device=dict(cuda_id=0,pci="0000:07:00.0"),
                        explicit_device=dict(cuda_id=0,pci="0000:07:00.0"),
                        native_selected_device=dict(cuda_id=0,pci="0000:07:00.0",ray_tracing=True))
        worker=dict(id=job["id"],scientific_identity=job["scientific_identity"],status="complete",
                    host="gpu01",gpu_index=1,gpu_uuid="GPU-assigned",command=["python3","frozen.py"],
                    started=start.isoformat(),finished=(start+timedelta(seconds=60)).isoformat(),
                    spec_sha256="incoming-hash",exit_code=0)
        claim=copy.deepcopy(worker);claim.update(job_sha256="job-hash",worker_result=copy.deepcopy(worker))
        return [job,scientific,worker,claim,copy.deepcopy(job),PREWARM,dict(job="job-hash",incoming="incoming-hash")]
    def test_valid_is_diagnostic_only(self):
        r=assess(*self.fixture());self.assertTrue(r["GPU_diagnostic_validated"],r)
        self.assertFalse(r["full_release_allowed"]);self.assertFalse(r["native_L2_L3_passed"])
    def test_missing_never_passes(self):
        f=self.fixture();f[1]=None;r=assess(*f);self.assertEqual(r["assessment"],"pending")
    def test_both_wrong_registered_identity(self):
        f=self.fixture();f[2]["scientific_identity"]=f[3]["scientific_identity"]="other/attempt1"
        self.assertFalse(assess(*f)["GPU_diagnostic_validated"])
    def test_repeated_samples_rejected(self):
        f=self.fixture();f[1]["samples"][1]["time"]=f[1]["samples"][0]["time"]
        self.assertFalse(assess(*f)["GPU_diagnostic_validated"])
    def test_old_samples_rejected(self):
        f=self.fixture();f[1]["samples"][0]["time"]="2026-10-20T00:00:00+00:00"
        self.assertFalse(assess(*f)["GPU_diagnostic_validated"])
    def test_exit_crash_log_rejects_success(self):
        f=self.fixture();f[5]+=" SIGSEGV";self.assertFalse(assess(*f)["GPU_diagnostic_validated"])
    def test_incoming_hash_rejected(self):
        f=self.fixture();f[6]["incoming"]="different";self.assertFalse(assess(*f)["GPU_diagnostic_validated"])
    def test_claim_embedded_worker_rejected(self):
        f=self.fixture();f[3]["worker_result"]["exit_code"]=1
        self.assertFalse(assess(*f)["GPU_diagnostic_validated"])
    def test_worker_running_is_pending(self):
        f=self.fixture();f[2]["status"]=f[3]["status"]="running";f[3]["worker_result"]=copy.deepcopy(f[2])
        r=assess(*f);self.assertFalse(r["GPU_diagnostic_validated"]);self.assertFalse(r["terminal_agreement"])
    def test_failed_peer_retained(self):
        f=self.fixture();f[1]["status"]=f[2]["status"]=f[3]["status"]="failed"
        f[2]["exit_code"]=1;f[3]["worker_result"]=copy.deepcopy(f[2])
        f[1]["samples"][-1]["contexts"].append(dict(uuid="GPU-peer"))
        r=assess(*f);self.assertTrue(r["terminal_agreement"],r);self.assertIsNotNone(r["first_sampled_peer"])
        self.assertFalse(r["GPU_diagnostic_validated"])
    def test_pci_mismatch_rejected(self):
        f=self.fixture();f[1]["explicit_device"]["pci"]="0000:08:00.0"
        self.assertFalse(assess(*f)["GPU_diagnostic_validated"])
    def test_log_prewarmer_required(self):
        f=self.fixture();f[5]="";self.assertFalse(assess(*f)["GPU_diagnostic_validated"])

if __name__ == "__main__": unittest.main()
