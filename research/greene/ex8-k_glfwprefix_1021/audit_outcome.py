"""CPU-only independent native outcome chain; stdout only, no native imports."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

PHASES = ["before_prewarm", "after_prewarm", "after_sapien_import",
          "after_explicit_Device", "after_helper_load", "after_native_glfwInit", "after_native_glfwVulkanSupported"]
FATAL = ["Traceback (most recent call last):", "SIGSEGV", "Segmentation fault",
         "CUDA/Curobo prewarm failed:", "planner fallback engaged:"]
PREWARM = "prewarmed CUDA/Curobo before SAPIEN import"

def stamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result

def assess(job, scientific, worker, claim, incoming, native_log, hashes):
    out = dict(assessment="pending", terminal_agreement=False,
               GPU_diagnostic_validated=False, native_L2_L3_passed=False,
               full_release_allowed=False, full_paper_reproduced=False,
               errors=[], first_sampled_peer=None)
    errors = out["errors"]
    if not all([scientific, worker, claim, incoming]):
        out["missing"] = [k for k,v in dict(scientific=scientific, worker=worker,
                                            claim=claim, incoming=incoming).items() if not v]
        return out
    for label, record in [("worker", worker), ("claim", claim)]:
        for key in ["id", "scientific_identity"]:
            if record.get(key) != job.get(key): errors.append(label+" registered "+key+" mismatch")
    if incoming != job: errors.append("incoming descriptor differs from registered job")
    for field in ["gpu_uuid", "gpu_index", "host"]:
        if worker.get(field) is None or worker.get(field) != claim.get(field):
            errors.append("worker/claim "+field+" mismatch")
    if claim.get("worker_result") != worker: errors.append("claim embedded worker differs")
    if claim.get("job_sha256") != hashes["job"]: errors.append("claim registered-job hash mismatch")
    if worker.get("spec_sha256") != hashes["incoming"]: errors.append("worker incoming-spec hash mismatch")
    command = job.get("host_commands", {}).get(worker.get("host"), {}).get("argv")
    if worker.get("command") != command: errors.append("worker command mismatch")
    statuses = dict(scientific=scientific.get("status"), worker=worker.get("status"), claim=claim.get("status"))
    out["statuses"] = statuses
    samples = scientific.get("samples", [])
    assigned = worker.get("gpu_uuid")
    for sample in samples:
        if any(c.get("uuid") != assigned for c in sample.get("contexts", [])):
            out["first_sampled_peer"] = sample
            break
    out["phase_counts"] = dict(Counter(s.get("phase") for s in samples))
    try:
        started, finished = stamp(worker["started"]), stamp(worker["finished"])
        times = [stamp(s["time"]) for s in samples]
        if not started < finished: errors.append("invalid worker lifetime")
        if any(b <= a for a,b in zip(times, times[1:])): errors.append("duplicate or reversed sample timestamps")
        if any(t < started or t > finished for t in times): errors.append("sample outside worker lifetime")
    except (ValueError, KeyError, TypeError):
        if worker.get("status") in ["complete", "failed"]:
            errors.append("missing or invalid sample/worker timestamp")
    out["native_log_fatal_markers"] = [x for x in FATAL if x in native_log]
    out["sampling_limit"] = "Labels precede inventory calls; exact cause and between-sample absence remain unproven."
    if statuses["scientific"] == statuses["worker"] == statuses["claim"] == "failed":
        out["terminal_agreement"] = not errors
        out["assessment"] = "retained_failed_diagnostic" if not errors else "inconsistent_failed_evidence"
        return out
    if statuses != dict(scientific="glfw_prefix_passed", worker="complete", claim="complete"):
        out["assessment"] = "pending_or_inconsistent_terminal_evidence"
        return out
    if worker.get("exit_code") != 0: errors.append("nonzero worker exit")
    if out["first_sampled_peer"]: errors.append("peer GPU context sampled")
    if scientific.get("monitor_errors"): errors.append("monitor errors")
    if any(out["phase_counts"].get(p, 0) < 3 for p in PHASES): errors.append("insufficient phase samples")
    if not any(c.get("uuid") == assigned for s in samples for c in s.get("contexts", [])):
        errors.append("no assigned context witness")
    if PREWARM not in scientific.get("prewarm_stdout", "") or PREWARM not in native_log:
        errors.append("missing independent prewarm corroboration")
    if out["native_log_fatal_markers"]: errors.append("native log has fatal markers")
    device=scientific.get("discovered_device", {})
    explicit=scientific.get("explicit_device", {})
    if not (device.get("cuda_id")==explicit.get("cuda_id")==0 and device.get("pci") and device.get("pci")==explicit.get("pci")):
        errors.append("explicit device inconsistency")
    if scientific.get("glfw_init")!=1 or scientific.get("glfw_vulkan_supported")!=1:
        errors.append("native GLFW prefix did not succeed")
    if scientific.get("scored_episodes") != 0: errors.append("diagnostic unexpectedly scores episodes")
    out["terminal_agreement"] = not errors
    out["GPU_diagnostic_validated"] = not errors
    out["assessment"] = "diagnostic_only_passed" if not errors else "success_evidence_rejected"
    return out

def audit(runtime, job_id, experiment):
    directory = runtime / "nightly-gpu"
    paths = dict(job=directory/"jobs"/(job_id+".json"),
                 incoming=directory/"incoming"/(job_id+".json"),
                 worker=directory/"runs"/job_id/"result.json",
                 claim=directory/"claims"/(job_id+".json"),
                 scientific=runtime/experiment/"result.json",
                 native_log=directory/"runs"/job_id/"run.log")
    observed, records, hashes = {}, {}, {}
    def read(path):
        if not path.exists(): return None
        raw = path.read_bytes(); observed[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw
    for label,path in paths.items():
        raw = read(path)
        hashes[label] = observed.get(str(path))
        records[label] = raw.decode(errors="replace") if label == "native_log" and raw is not None else json.loads(raw) if raw else None
    if not records["job"]: raise ValueError("registered job missing")
    job = records["job"]
    protocol_path = Path(job["authorization_evidence"])
    protocol = json.loads(read(protocol_path))
    integrity_errors = []
    if observed[str(protocol_path)] != job["authorization_sha256"]:
        integrity_errors.append("authorization hash mismatch")
    for name,digest in protocol["input_sha256"].items():
        read(Path(name))
        if observed.get(name) != digest: integrity_errors.append("frozen input mismatch: "+name)
    out = assess(job, records["scientific"], records["worker"], records["claim"],
                 records["incoming"], records["native_log"] or "", hashes)
    changed = [name for name,digest in observed.items() if not Path(name).exists()
               or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    out.update(integrity_errors=integrity_errors, changed_during_read=changed,
               frozen_inputs_verified=len(protocol["input_sha256"]), evidence_sha256=observed,
               job_id=job_id, scientific_identity=job["scientific_identity"])
    if integrity_errors or changed:
        out.update(GPU_diagnostic_validated=False, terminal_agreement=False,
                   assessment="input_integrity_or_concurrent_change_rejected")
    return out

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, default=Path("/home/zifanz4/openwam-runtime"))
    p.add_argument("--job", default="ex8k-native-glfw-prefix")
    p.add_argument("--experiment", default="ex8-k_glfwprefix_1021")
    a=p.parse_args()
    print(json.dumps(audit(a.runtime,a.job,a.experiment),indent=2))
