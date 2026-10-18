"""Freeze a scheduling-only derivative; never modify the active driver."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

SOURCE=Path(__file__).resolve().parent
ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
DEST=ROOT/'resource-handoff'
assert not DEST.exists()
DEST.mkdir()
old=(ROOT/'code/run_lane.py').read_text()
new=old
replacements={
    'def main(gpu,variant):':'def main(gpu,variant,seed,allocated_seconds):\n    assert seed in [43,44] and 0<allocated_seconds<=10800\n    P["seeds"]=[seed]\n    P["lane_active_seconds"]=allocated_seconds',
    'assert gpu in [3,4] and variant in P[\'variants\']':'assert gpu in [1,2,3,4] and variant in P[\'variants\']',
    "state=ROOT/f'lane-gpu{gpu}.json'":"state=ROOT/f'lane-gpu{gpu}-{variant}-seed{seed}.json'",
    "lock=(R/f'gpu-{gpu}-ex4f.lock').open('w')":"lock=(R/f'gpu-{gpu}-ex4f-stage2.lock').open('w')",
    'port=19940+gpu':'port=20000+10*gpu+seed',
    "p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',required=True);a=p.parse_args();main(a.gpu,a.variant)":
        "p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--allocated-seconds',type=int,required=True);a=p.parse_args();main(a.gpu,a.variant,a.seed,a.allocated_seconds)"
}
for before,after in replacements.items():
    assert new.count(before)==1,before
    new=new.replace(before,after)
ast.parse(new)
(DEST/'run_cell.py').write_text(new)
(DEST/'scheduling-only.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/run_lane.py',tofile='handoff/run_cell.py')))
for name in ['RESOURCE_HANDOFF.md','handoff_supervisor.py','prepare_handoff.py']:
    shutil.copy2(SOURCE/name,DEST/name)
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.is_file()}
(DEST/'freeze.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
    files=files,original_driver_sha256=hashlib.sha256(old.encode()).hexdigest(),scope='Scheduling only; no active source snapshot edited.'),indent=2)+'\n')
print(DEST)
