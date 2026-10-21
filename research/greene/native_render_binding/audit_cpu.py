"""CPU-only provenance and candidate gate checks. Never imports GPU packages."""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

R=Path("/home/zifanz4/openwam-runtime")
ROOT=R/"ex8-a_officialrepro_1021"
B=Path(__file__).parent
S=R/"benchmark-env/lib/python3.10/site-packages/sapien"

def main(output):
    paths=[Path("/home/zifanz4/.local/state/openwam-selfcheck/paused.json")]+[R/n/"paused.json" for n in ["ex8-a_officialrepro_1021","ex8-b_candidatefilter_1021","ex8-d_nativefull_1021","ex8-e_rowrecovery_1021"]]
    assert not any(p.exists() for p in paths)
    files=[ROOT/"official/benchmarks/robotwin/eval_policy_wrapper.py",ROOT/"RoboTwin/envs/_base_task.py",S/"wrapper/scene.py",S/"wrapper/renderer.py",S/"pysapien/render.pyi",S/"pysapien/__init__.pyi",R/"ex8-d_nativefull_1021/code/run_full.py",R/"ex8-d_nativefull_1021/code/classify_log.py",R/"nightly-gpu/code-v3/probe.py",B/"adapter.py",B/"probe_candidate.py",B/"test_adapter.py",B/"PLAN.md",Path(__file__)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source=files[0].read_text();tree=ast.parse(source)
    bootstrap=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="bootstrap_robotwin_module")
    calls=[(n.lineno,n.func.id) for n in ast.walk(bootstrap) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert min(n for n,k in calls if k=="_prewarm_cuda_for_curobo") < min(n for n,k in calls if k=="_load_robotwin_eval_module")
    renderer=ast.parse((S/"wrapper/renderer.py").read_text());cls=next(n for n in renderer.body if isinstance(n,ast.ClassDef) and n.name=="SapienRenderer")
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    super_call=[n for n in ast.walk(init) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="__init__"]
    assert len(super_call)==1 and not super_call[0].args and not super_call[0].keywords
    assert "sapien.render.RenderSystem()" in (S/"wrapper/scene.py").read_text()
    frozen=json.loads((R/"ex8-d_nativefull_1021/protocol.json").read_text())
    for p in [R/"ex8-d_nativefull_1021/code/run_full.py",R/"ex8-d_nativefull_1021/code/classify_log.py"]:assert hashes[str(p)]==frozen["input_sha256"][str(p)]
    sync=json.loads((R/"official-sync/latest.json").read_text());assert sync["passed"] and sync["before"]==sync["after"]==frozen["official_sync"]["after"]
    recovery=json.loads((R/"ex8-e_rowrecovery_1021/protocol.json").read_text())
    failed=json.loads((R/"nightly-gpu/runs/ex8d-full-002-blocks-ranking-rgb/result.json").read_text())
    import math
    occupied=math.ceil(failed["occupied_parent_seconds"])
    combined=recovery["combined_reserved_seconds"]+occupied
    assert combined==1007796
    probe_seconds=180;assert combined+probe_seconds<=280*3600
    result=subprocess.run([sys.executable,"-m","unittest","-v","test_adapter.py"],cwd=B,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    report={"created":dt.datetime.now(dt.timezone.utc).isoformat(),"kind":"CPU_NATIVE_RENDER_BINDING_PREPARATION","L0_static_passed":True,"L1_CPU_controls_passed":True,"L1_GPU_renderer_probe_passed":False,"L2_L3_native_episode_validation_passed":False,"dispatch_allowed":False,"new_gpu_hours":0,"official_source_compatibility":"Unchanged official tracking commit; frozen runtimes unchanged","official_sync":sync,"input_sha256":hashes,"tests":result.stderr,"findings":["Legacy SapienRenderer silently drops device kwargs","Default Scene uses RenderSystem() without explicit device","Native bootstrap prewarms before loading RoboTwin; adapter retains native main and argv","Explicit renderer priming remains an unproven candidate; matching PCI is insufficient without all-context monitoring"],"cost":{"conditional_recovery_reserved_seconds":combined,"proposed_probe_cap_seconds":probe_seconds,"proposed_reserved_gpu_hours":(combined+probe_seconds)/3600,"remaining_headroom_seconds":280*3600-combined-probe_seconds,"existing_caps_unchanged":True},"next_gates":["Freeze a separate <=180s renderer probe and exact input hashes; do not dispatch draft","Use existing dispatcher after idle30min; avoid colliding with active native peer jobs","Require actual default-scene device match and no sampled peer contexts; retain monitor limits","Then validate native rollout and freeze separate row recovery preserving failed row and23 excluded original partial episodes"]}
    assert not any(p.exists() for p in paths)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ["input_sha256","tests","official_sync"]},indent=2))

if __name__=="__main__":main(Path(sys.argv[1]))
