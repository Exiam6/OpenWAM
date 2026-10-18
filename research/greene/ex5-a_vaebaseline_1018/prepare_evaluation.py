"""Freeze unchanged corrected evaluator inputs; no model or simulator launch."""
import json
from pathlib import Path
import shutil
import datetime as dt
from build_cache import sha

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
G=ROOT.parent/'ex4-g_calibratedfactorial_1017'
assert not (ROOT/'gpu-registration.json').exists(),'Preparation must precede GPU admission'
assert not (ROOT/'evaluation-template').exists(),'Preserve previous preparation'
source=G/'template-code/fixed';dest=ROOT/'evaluation-template';shutil.copytree(source,dest)
for name in ['base-environment.json','historical-outcomes.json']:shutil.copy2(G/name,ROOT/name)
shutil.copytree(G/'robot-configs',ROOT/'robot-configs',symlinks=False)
p=json.loads((G/'registration.json').read_text());files={}
for x in dest.iterdir():
    if x.is_file():assert sha(x)==sha(source/x.name);files[str(x)]=sha(x)
for x in (ROOT/'robot-configs').rglob('*'):
    if x.is_file():assert sha(x)==sha(G/'robot-configs'/x.relative_to(ROOT/'robot-configs'));files[str(x)]=sha(x)
for name in ['base-environment.json','historical-outcomes.json']:files[str(ROOT/name)]=sha(ROOT/name)
record=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,source=str(source),files=files,scenes=p['scenes'],seeds=p['seeds'],
    variant='vae48_native',pilot_per_seed={t:v[:5] for t,v in p['scenes'].items()},remaining_per_seed={t:v[5:] for t,v in p['scenes'].items()},
    planned_evaluations=180,scene_count=60,training_seeds=3,scientific_evaluator_modified=False,
    note='Unchanged corrected native evaluator, action interface, task limits, renderer and calibration; only the route label/checkpoint differs. All inputs frozen before GPU work.')
(ROOT/'evaluation-template-audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(passed=True,frozen_files=len(files),planned_evaluations=180)))
