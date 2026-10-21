import copy, unittest
from outcome import audit
class Evidence(unittest.TestCase):
 def base(self):
  phases=["before_prewarm","after_prewarm","after_sapien_import","after_explicit_Device","after_helper_load","after_native_Context_Create","after_native_createResourceManager","after_explicit_RenderSystem"]
  s=dict(status="context_split_passed",samples=[dict(phase=p,contexts=[dict(uuid="GPU-assigned")]) for p in phases for _ in range(3)],prewarm_stdout="prewarmed CUDA/Curobo before SAPIEN import",discovered_device=dict(cuda_id=0,pci="0000:07:00.0"),explicit_device=dict(cuda_id=0,pci="0000:07:00.0"))
  w=dict(id="probe",status="complete",gpu_uuid="GPU-assigned",exit_code=0);c=dict(w)
  return s,w,c
 def test_pending_no_conclusion(self):
  r=audit(None,None,None);self.assertFalse(r["GPU_diagnostic_validated"]);self.assertFalse(r["terminal_agreement"])
 def test_complete_never_releases_full(self):
  r=audit(*self.base());self.assertTrue(r["GPU_diagnostic_validated"]);self.assertFalse(r["full_release_allowed"])
 def test_partial_samples_reject(self):
  s,w,c=self.base();s["samples"].pop();self.assertFalse(audit(s,w,c)["GPU_diagnostic_validated"])
 def test_peer_rejects_prose_success(self):
  s,w,c=self.base();s["samples"][-1]["contexts"].append(dict(uuid="GPU-peer"));self.assertFalse(audit(s,w,c)["GPU_diagnostic_validated"])
 def test_terminal_disagreement(self):
  s,w,c=self.base();c["status"]="running";self.assertFalse(audit(s,w,c)["terminal_agreement"])
 def test_identity_mismatch(self):
  s,w,c=self.base();c["id"]="different";self.assertFalse(audit(s,w,c)["GPU_diagnostic_validated"])
 def test_failed_prewarm_rejects_success(self):
  s,w,c=self.base();s["prewarm_stdout"]+="CUDA/Curobo prewarm failed:";self.assertFalse(audit(s,w,c)["GPU_diagnostic_validated"])
 def test_monitor_error_rejects_prose_success(self):
  s,w,c=self.base();s["monitor_errors"]=["incomplete inventory"];self.assertFalse(audit(s,w,c)["GPU_diagnostic_validated"])
 def test_wrong_pci_rejects(self):
  s,w,c=self.base();s["explicit_device"]["pci"]="0000:08:00.0";self.assertFalse(audit(s,w,c)["GPU_diagnostic_validated"])
 def test_failed_keeps_observed_peer(self):
  s,w,c=self.base();s["status"]=w["status"]=c["status"]="failed";w["exit_code"]=1;s["samples"][-1]["contexts"].append(dict(uuid="GPU-peer"));r=audit(s,w,c);self.assertTrue(r["terminal_agreement"]);self.assertIsNotNone(r["first_sampled_peer"]);self.assertFalse(r["GPU_diagnostic_validated"])
if __name__=="__main__":unittest.main()
