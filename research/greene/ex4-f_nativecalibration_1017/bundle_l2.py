"""Freeze the complete seed42 adjust control, with all ten paired videos."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
SOURCE=Path(__file__).resolve().parent
P=json.loads((ROOT/'registration.json').read_text())
baseline=json.loads((ROOT/'baseline-selected.json').read_text())
audit=json.loads((ROOT/'independent-result-audit.json').read_text())
media=json.loads((ROOT/'paired-videos-adjust_bottle-seed42.json').read_text())['records']
assert len(media)==10
lines=['# Restored calibration improves the seed42 adjust pilot','',
    'On the fixed first five accepted adjust-bottle scenes, both endpoint policies improve after restoring native planner calibration. '
    'This is one training seed and one task; handover, place and the other training seeds remain pending.', '',
    'Next: finish the registered three-seed pilot under the unchanged compute cap. '
    'Do not interpret this control-task result as a representation ranking.', '',
    '| Model | Old path | Restored native calibration | Paired gains | Paired losses |', '|---|---:|---:|---:|---:|']
data=[]
for variant in P['variants']:
    cell=f'{variant}-seed42';path=ROOT/'evaluation'/cell/'fixed/adjust_bottle/summary.json';d=json.loads(path.read_text())
    assert d['complete'] and d['episodes']==5
    old={x['scene_seed']:x for x in baseline if x['route']==variant and x['training_seed']==42 and x['task']=='adjust_bottle'}
    gains=sum(x['success'] and not old[x['scene_seed']]['success'] for x in d['records'])
    losses=sum(not x['success'] and old[x['scene_seed']]['success'] for x in d['records'])
    old_successes=sum(x['success'] for x in old.values())
    lines.append(f"| {variant} | {old_successes/5:.0%} ({old_successes}/5) | {d['successes']/5:.0%} ({d['successes']}/5) | {gains} | {losses} |")
    for row in d['records']:
        check=next(x for x in audit['rows'] if x['cell']==cell and x['condition']=='fixed' and x['task']=='adjust_bottle' and x['scene_seed']==row['scene_seed'])
        assert check['verified'] and check['paired_first_chunk_max_action_difference']==0
        data.append(dict(cell=cell,scene_seed=row['scene_seed'],old=old[row['scene_seed']],fixed=row,audit=check))
lines+=['','The background question is whether execution failure obscures learned-policy performance. '
    'The single change preserves the `aloha-agilex` filename marker that selects native arm-base calibration. '
    'Checkpoint, prompts, random seeds, initial scene, cameras, action horizon and success rule are held fixed. '
    'Initial proprioception, all three camera views and the first eight actions match the historical control exactly in all ten pairs.', '',
    'The preregistered easy-task gate requires at least one success per model and no infrastructure failures before advancing. '
    'Both models pass. This gate establishes a working policy-executor combination; the primary handover hypothesis is still pending.', '',
    '## Every paired clip', '',
    'Old is above; restored calibration is below. The three original camera views are shown from left to right: head, left wrist, right wrist. '
    'Clips align action index at12fps, not physical time; successful shorter episodes hold their final frame. '
    'All clips fully decode. Selected start/intermediate/final contact sheets were inspected; this is not a claim of manual frame-by-frame review. '
    'No external full-robot camera was added.', '',
    '| Model | Scene | Old | Restored | Video |', '|---|---:|---|---|---|']
for x in media:
    relative=Path('videos')/x['cell']/f"seed-{x['scene_seed']}.mp4"
    lines.append(f"| {x['cell']} | {x['scene_seed']} | {'success' if x['old_success'] else 'failure'} | {'success' if x['fixed_success'] else 'failure'} | [Play / download]({relative}) |")
lines+=['','## Provenance and limits','',
    f"- Registration commit: `{P['commit']}`; source model commit: `{P['model_commit']}`.",
    '- `data.json` contains every original and corrected result plus independent checks. `registration.json` contains scene/checkpoint/code hashes.',
    '- Model names refer to raw768-dimensional versus compact48-dimensional frozen visual representations, with the same native action architecture.',
    '- Development pilot: first-five selection rule fixed before inspecting this subset; original historical outcomes were available. No unseen-test or cross-seed claim.',
    '- The native success rule is unchanged. Follow-up results must remain separate from this frozen control-task snapshot.']
readme='\n'.join(lines)+'\n'
output=ROOT/'L2-adjust-seed42-evidence.zip';assert not output.exists(),'Do not overwrite a frozen evidence bundle'
with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_STORED) as z:
    z.writestr('README.md',readme);z.writestr('data.json',json.dumps(data,indent=2)+'\n')
    z.write(ROOT/'registration.json','registration.json')
    z.write(ROOT/'budget-review-L2.json','budget-review-L2.json')
    for x in media:
        video=Path(x['video']);prefix=Path('videos')/x['cell']/f"seed-{x['scene_seed']}"
        for ext in ['.mp4','.jpg','.json']:z.write(video.with_suffix(ext),str(prefix)+ext)
with zipfile.ZipFile(output) as z:assert z.testzip() is None
record=dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,paired_episodes=len(data),videos=len(media),zip_crc_passed=True)
(ROOT/'L2-bundle.json').write_text(json.dumps(record,indent=2)+'\n')
# Repository report uses direct artifact paths instead of zip-relative links.
report=readme
for x in media:
    relative=f"videos/{x['cell']}/seed-{x['scene_seed']}.mp4"
    report=report.replace(f']({relative})',f"]({x['video']})")
report+=f'\n[Full-resolution evidence bundle]({output})\n'
(SOURCE/'L2_RESULTS.md').write_text(report);(ROOT/'L2_RESULTS.md').write_text(report)
print(json.dumps(record))
