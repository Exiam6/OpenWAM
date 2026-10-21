"""Read-only conservative audit of scientific result, dispatcher worker and claim."""
import argparse, collections, json
from pathlib import Path

def audit(scientific, worker, claim):
    statuses={"scientific": scientific.get("status") if scientific else None,
              "worker": worker.get("status") if worker else None,
              "claim": claim.get("status") if claim else None}
    out={"statuses":statuses,"terminal_agreement":False,"GPU_diagnostic_validated":False,
         "native_L2_L3_passed":False,"full_release_allowed":False,"full_paper_reproduced":False}
    if not scientific or not worker or not claim:
        return dict(out, assessment="pending evidence; no scientific conclusion")
    if worker.get("id")!=claim.get("id") or worker.get("gpu_uuid")!=claim.get("gpu_uuid"):
        return dict(out, assessment="worker/claim identity mismatch")
    assigned=claim.get("gpu_uuid")
    if not assigned:
        return dict(out, assessment="missing assigned GPU identity")
    samples=scientific.get("samples",[])
    peers=[s for s in samples if any(c.get("uuid")!=assigned for c in s.get("contexts",[]))]
    counts=dict(collections.Counter(s.get("phase") for s in samples))
    out.update(phase_sample_counts=counts, assigned_context_observed=any(c.get("uuid")==assigned for s in samples for c in s.get("contexts",[])),
               first_sampled_peer=peers[0] if peers else None,
               native_prewarm_succeeded="prewarmed CUDA/Curobo before SAPIEN import" in scientific.get("prewarm_stdout","") and "CUDA/Curobo prewarm failed:" not in scientific.get("prewarm_stdout",""))
    s,w,c=statuses.values()
    if s=="failed" and w==c=="failed" and worker.get("exit_code",0)!=0:
        out.update(terminal_agreement=True,assessment="retained failed diagnostic; sampled peer boundary" if peers else "retained failed diagnostic; inspect actual error")
    elif s=="context_split_passed" and w==c=="complete" and worker.get("exit_code")==0 and not peers and not scientific.get("monitor_errors"):
        required=["before_prewarm","after_prewarm","after_sapien_import","after_explicit_Device","after_helper_load","after_native_Context_Create","after_native_createResourceManager","after_explicit_RenderSystem"]
        sufficient=all(counts.get(p,0)>=3 for p in required)
        devices=[scientific.get(k,{}) for k in ["discovered_device","explicit_device"]]
        pci_ok=all(d.get("cuda_id")==0 and d.get("pci") for d in devices) and devices[0].get("pci")==devices[1].get("pci")
        passed=sufficient and pci_ok and out["native_prewarm_succeeded"] and out["assigned_context_observed"]
        out.update(terminal_agreement=True,GPU_diagnostic_validated=passed,assessment="diagnostic passed; native scene validation still required" if passed else "terminal success lacks required scientific evidence")
    else:
        out["assessment"]="pending or inconsistent terminal evidence"
    out["sampling_limit"]="Phase label captured before inventory call; a sample can span the next boundary. No exact internal cause or between-sample absence proven."
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--scientific",type=Path,required=True);p.add_argument("--worker",type=Path,required=True);p.add_argument("--claim",type=Path,required=True);a=p.parse_args()
    def read(path): return json.loads(path.read_text()) if path.exists() else None
    print(json.dumps(audit(read(a.scientific),read(a.worker),read(a.claim)),indent=2))
if __name__=="__main__": main()
