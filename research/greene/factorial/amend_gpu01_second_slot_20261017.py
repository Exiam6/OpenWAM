"""Amendment 4 (2026-10-17 ~02:55 CDT, before any learned policy was scored): a second post-worker slot per GPU.

Scheduling only. Each GPU may host two post-worker jobs (scene screening or evaluation cells). Slot 0 is
unchanged. Slot 1 uses its own lock file, state file, simulator runtime directory and a virtual
STUDY_GPU = gpu + 8 (so evaluate_cell.py's port base_port + STUDY_GPU never collides), keeps the same
physical GPU (CUDA_VISIBLE_DEVICES uuid, EXPECTED_RENDER_PCI), and is admitted when the GPU reports no
ECC errors and at least `shared_min_free_mib` free memory. Claims stay atomic, so no job runs twice.
No scientific field, scene rule, success rule or evaluation script changes.
"""
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

R = Path("/home/zifanz4/openwam-runtime/rae-factorial-20261010")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old, s.count(old))
    return s.replace(old, new)


def main():
    archive = R / "resource-amendment-20261017-second-slot"
    assert not archive.exists()
    assert not list((R / "post-claims").glob("evaluate-*.json")), "evaluation already started"
    archive.mkdir()
    for name in ["post_worker.py", "code-freeze.json", "eval-freeze.json"]:
        shutil.copy2(R / name, archive / name)
    s = (R / "post_worker.py").read_text()
    s = rep(s, "def main(gpu):\n", '''def slot_status(gpu, expected, policy):
    """Second-slot admission: same physical GPU, enough free memory, no ECC errors."""
    line = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total,pci.bus_id,memory.total",
         "--format=csv,noheader,nounits"], text=True, timeout=20)
    values = [x.strip() for x in line.strip().split(",")]
    if values[0] != expected:
        raise RuntimeError("GPU UUID mismatch")
    if values[3] != "0":
        raise RuntimeError("GPU reports uncorrected ECC errors")
    ok = int(values[5]) - int(values[1]) >= int(policy["shared_min_free_mib"])
    values.append("slot1-ok" if ok else "slot1-full")
    return ok, values


def main(gpu, slot=0):
''')
    s = rep(s, '    statefile = ROOT / f"post-worker-gpu{gpu}.json"\n',
            '    tag = gpu if slot == 0 else f"{gpu}-slot{slot}"\n    statefile = ROOT / f"post-worker-gpu{tag}.json"\n'
            '    status = (lambda g, u: gpu_status(g, u)) if slot == 0 else (lambda g, u: slot_status(g, u, policy))\n')
    s = rep(s, '            free, values = gpu_status(gpu, policy["gpus"][gpu])\n', '            free, values = status(gpu, policy["gpus"][gpu])\n')
    s = rep(s, '            with (ROOT.parent / f"gpu-{gpu}.lock").open("a") as lock:\n',
            '            with (ROOT.parent / (f"gpu-{gpu}.lock" if slot == 0 else f"gpu-{gpu}-slot{slot}.lock")).open("a") as lock:\n')
    s = rep(s, '                if not gpu_status(gpu, policy["gpus"][gpu])[0]:\n', '                if not status(gpu, policy["gpus"][gpu])[0]:\n')
    s = rep(s, '                    STUDY_GPU=gpu,\n', '                    STUDY_GPU=str(int(gpu) + 8 * slot),\n')
    s = rep(s, '                    ROBOTWIN_RUNTIME_ROOT=str(ROOT / f"simulator-runtime-gpu{gpu}"),\n',
            '                    ROBOTWIN_RUNTIME_ROOT=str(ROOT / f"simulator-runtime-gpu{tag}"),\n')
    s = rep(s, '    main(parser.parse_args().gpu)\n',
            '    parser.add_argument("--slot", type=int, choices=[0, 1], default=0)\n    args = parser.parse_args()\n    main(args.gpu, args.slot)\n')
    s = rep(s, '                row = dict(job, pid=os.getpid(), gpu=gpu, status="running", started=now().isoformat())\n',
            '                row = dict(job, pid=os.getpid(), gpu=gpu, slot=slot, status="running", started=now().isoformat())\n')
    compile(s, "post_worker.py", "exec")
    (R / "post_worker.py").write_text(s)
    for name in ["code-freeze.json", "eval-freeze.json"]:
        m = json.loads((R / name).read_text())
        m["files"][str(R / "post_worker.py")] = sha(R / "post_worker.py")
        m["resource_amendment_second_slot"] = str(archive)
        (R / name).write_text(json.dumps(m, indent=2) + "\n")
    rec = dict(applied_at=dt.datetime.now(dt.timezone.utc).isoformat(), change="second post-worker slot per GPU (scheduling only)",
               post_worker_sha256=sha(R / "post_worker.py"), archived=str(archive), learned_policy_scored=False)
    (R / "resource-amendment-gpu01-second-slot.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
