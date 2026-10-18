"""Archive a descriptive one-scene observation, distinct from the registered pilot verdict."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
SOURCE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    root=ROOT/'handover-first-scene';data=json.loads((root/'data.json').read_text())
    media=json.loads((root/'paired-videos.json').read_text())['records'];assert len(media)==6
    review=json.loads((root/'visual-review.json').read_text());assert review['figure_inspected'] and review['selected_contactsheets_inspected']
    lines=['# Tracking improves, but the first handover scene still receives no closing command','',
        'On scene6201000, all six fixed-endpoint policies fail under both the old and restored native executor. '
        'Restored calibration reduces median target-to-next-observation position error for each arm of each policy. '
        'Nevertheless, none of the corrected trajectories commands either gripper below0.5 during its800 action steps.', '',
        'This is a post-hoc description of one shared scene across two models and three training seeds, not a separate hypothesis test. '
        'It identifies a concrete remaining behavior after the execution fix; it does not prove why the policy chooses that behavior. '
        'Other scenes may issue closing commands. The full pilot verdict remains pending in this snapshot.', '',
        'Next: finish all fixed pilot scenes and the registered architecture comparison. '
        'Keep policy-command diagnostics separate from expert replay success and from task-level success rates.', '',
        '![Native camera frames, commands and tracking errors](control-observation.png)','',
        '| Checkpoint | Left error: old → fixed (mm) | Right error: old → fixed (mm) | Corrected closing commands (left/right) |','|---|---:|---:|---:|']
    for row in data['rows']:
        c=row['conditions'];old=c['old']['stats'];new=c['fixed']['stats']
        for condition in ['old','fixed']:
            d=Path(c[condition]['directory']);assert sha(d/'result.json')==c[condition]['result_sha256'];assert sha(d/'trace.jsonl')==c[condition]['trace_sha256']
        assert all(new[s]['median_position_error_mm']<old[s]['median_position_error_mm'] for s in ['left','right'])
        lines.append(f"| {row['cell']} | {old['left']['median_position_error_mm']:.2f} → {new['left']['median_position_error_mm']:.2f} | {old['right']['median_position_error_mm']:.2f} → {new['right']['median_position_error_mm']:.2f} | {new['left']['closed_command_steps']}/{new['right']['closed_command_steps']} |")
    lines+=['','The tracking comparison uses the six newly rerun old-path controls. '
        'The videos below pair corrected episodes with their original historical old-path episodes; these are explicitly different old recordings. '
        'All controls use the same checkpoint, initial scene, native inference settings and task success rule. '
        'After different actions execute, trajectories diverge; the median tracking differences are descriptive, not matched-state causal estimates.', '',
        '## All six paired videos','',
        'Old above, corrected below; head, left wrist and right wrist views. Action-index alignment at12fps; shorter recordings hold their last frame. '
        'Every clip fully decodes. The complete figure and two seed43 contact sheets were visually inspected at sampled start/intermediate/final frames. '
        'Native cameras do not guarantee a complete external view of the robot, and wrist images may be occluded.', '',
        '| Checkpoint | Video |','|---|---|']
    for x in media:
        assert x['full_decode_passed'] and sha(x['video'])==x['video_sha256']
        lines.append(f"| {x['cell']} | [Play / download](videos/{x['cell']}.mp4) |")
    readme='\n'.join(lines)+'\n';out=ROOT/'handover-first-scene-evidence.zip';assert not out.exists()
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.writestr('README.md',readme)
        for name in ['data.json','paired-videos.json','visual-review.json','control-observation.png','control-observation.pdf']:z.write(root/name,name)
        for name in ['plot_handover_control.py','bundle_handover_control.py','make_pairs.py']:z.write(SOURCE/name,'source/'+name)
        for row in data['rows']:
            for condition in ['old','fixed']:
                d=Path(row['conditions'][condition]['directory'])
                for name in ['result.json','trace.jsonl']:z.write(d/name,f"traces/{row['cell']}/{condition}/{name}")
        for x in media:
            video=Path(x['video'])
            for ext in ['.mp4','.jpg','.json']:z.write(video.with_suffix(ext),f"videos/{x['cell']}{ext}",compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    record=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),path=str(out),sha256=sha(out),bytes=out.stat().st_size,videos=6,traces=12,zip_crc_passed=True)
    (root/'bundle.json').write_text(json.dumps(record,indent=2)+'\n')
    report=readme.replace('](control-observation.png)',f']({root}/control-observation.png)')
    for x in media:report=report.replace(f"](videos/{x['cell']}.mp4)",f"]({x['video']})")
    report+=f'\n[Full-quality evidence bundle]({out})\n'
    for p in [root/'RESULTS.md',SOURCE/'HANDOVER_FIRST_SCENE.md']:p.write_text(report)
    print(json.dumps(record))

if __name__=='__main__':main()
