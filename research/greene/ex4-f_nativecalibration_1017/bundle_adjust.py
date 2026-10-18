"""Freeze the complete three-seed adjust control without replacing the earlier snapshot."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
SOURCE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=json.loads((ROOT/'registration.json').read_text())
    baseline=json.loads((ROOT/'baseline-selected.json').read_text())
    audit=json.loads((ROOT/'independent-result-audit.json').read_text())
    review=json.loads((ROOT/'adjust-three-seed-visual-review.json').read_text())
    assert review['selected_contactsheets_inspected']
    media=[]
    for seed in p['seeds']:
        rows=json.loads((ROOT/f'paired-videos-adjust_bottle-seed{seed}.json').read_text())['records']
        assert len(rows)==10;media.extend(rows)
    assert len({(x['cell'],x['scene_seed']) for x in media})==30
    data=[];summaries=[]
    for variant in p['variants']:
        for seed in p['seeds']:
            cell=f'{variant}-seed{seed}'
            summary=json.loads((ROOT/'evaluation'/cell/'fixed/adjust_bottle/summary.json').read_text())
            assert summary['complete'] and summary['episodes']==5 and summary['technical_failures']==0
            old={x['scene_seed']:x for x in baseline if x['route']==variant and x['training_seed']==seed and x['task']=='adjust_bottle'}
            assert set(old)==set(p['selected']['adjust_bottle'])
            gain=loss=0
            for result in summary['records']:
                scene=result['scene_seed'];before=old[scene]
                check=next(x for x in audit['rows'] if (x['cell'],x['condition'],x['task'],x['scene_seed'])==(cell,'fixed','adjust_bottle',scene))
                assert check['verified'] and check['paired_first_chunk_max_action_difference']==0
                assert check['initial_state_exact'] and all(x==0 for x in check['all_three_camera_mae'].values())
                video=next(x for x in media if x['cell']==cell and x['scene_seed']==scene)
                result_path=ROOT/'evaluation'/cell/'fixed/adjust_bottle'/f'seed-{scene}'/'result.json'
                assert video['new_result_sha256']==sha(result_path) and video['video_sha256']==sha(video['video']) and video['full_decode_passed']
                gain+=result['success'] and not before['success'];loss+=before['success'] and not result['success']
                data.append(dict(cell=cell,scene_seed=scene,old=before,fixed=result,audit=check,media=video))
            summaries.append(dict(variant=variant,seed=seed,n=5,old_successes=sum(x['success'] for x in old.values()),fixed_successes=summary['successes'],gains=gain,losses=loss))
    lines=['# Native calibration recovers the adjust-bottle pilot across three training seeds','',
        'The completed adjust-bottle control tests whether restoring native planner calibration improves an already working task. '
        'It covers the same five frozen scene IDs for each of two policies and three training seeds; this is a development subset, not 30 independent scene IDs.', '',
        'The single execution change preserves the ALOHA configuration marker. Final step20000 checkpoints, native inference, prompts, RNG, initial scene, camera views and success rule remain paired. '
        'This comparison estimates the execution change within each model; it does not establish a representation advantage.', '',
        '| Model | Training seed | Old execution | Native calibration | Gains | Losses |','|---|---:|---:|---:|---:|---:|']
    for x in summaries:
        lines.append(f"| {x['variant']} | {x['seed']} | {x['old_successes']/x['n']:.0%} ({x['old_successes']}/{x['n']}) | {x['fixed_successes']/x['n']:.0%} ({x['fixed_successes']}/{x['n']}) | {x['gains']} | {x['losses']} |")
    lines+=['','The preregistered seed42 easy-task gate requires at least one success per model with no infrastructure failure. '
        'Both models pass; seeds43/44 provide replication on the same development scenes. The primary handover hypothesis and the full pilot remain pending in this frozen snapshot. '
        'Next: complete handover and place, then use the separately registered corrected factorial to compare architectures.', '',
        'All30 pairs pass independent checks: exact initial state, all three initial camera errors zero, finite action traces and identical first native action chunks. '
        'Every video fully decodes. Four newly inspected contact sheets cover start, intermediate and final samples for both models at seeds43/44; the prior seed42 review remains separate. '
        'These are sampled visual checks, not manual inspection of every frame.', '',
        'Videos show the actual head and two wrist policy views, old above and corrected below. '
        'Native wrist images are often blank or occluded; the lifted bottle can approach the head-view boundary. No extra external full-robot view was recorded. '
        'Alignment uses action index at12fps, not physical time; a shorter episode holds its final frame.', '',
        '## All paired clips','', '| Model and seed | Scene | Old outcome | Corrected outcome | Video |','|---|---:|---|---|---|']
    for x in media:
        lines.append(f"| {x['cell']} | {x['scene_seed']} | {'success' if x['old_success'] else 'failure'} | {'success' if x['fixed_success'] else 'failure'} | [Play / download](videos/{x['cell']}/seed-{x['scene_seed']}.mp4) |")
    lines+=['','## Provenance','',f"Registration: `{p['commit']}`; model source: `{p['model_commit']}`.",
        'The bundle contains all30 paired clips, their contact sheets and metadata, per-episode results and audits, the registration, and the sampled visual review. '
        'Historical failed outcomes and the earlier seed42 evidence bundle are retained. Subsequent results do not alter this snapshot.']
    readme='\n'.join(lines)+'\n';output=ROOT/'adjust-three-seed-evidence.zip';assert not output.exists()
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_STORED) as z:
        z.writestr('README.md',readme);z.writestr('data.json',json.dumps(dict(summaries=summaries,episodes=data),indent=2)+'\n')
        for name in ['registration.json','adjust-three-seed-visual-review.json']:z.write(ROOT/name,name)
        for name in ['bundle_adjust.py','make_pairs.py','audit_results.py','PLAN.md','registry.json']:z.write(SOURCE/name,'source/'+name)
        for x in media:
            video=Path(x['video']);prefix=Path('videos')/x['cell']/f"seed-{x['scene_seed']}"
            for ext in ['.mp4','.jpg','.json']:z.write(video.with_suffix(ext),str(prefix)+ext)
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    record=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),path=str(output),sha256=sha(output),bytes=output.stat().st_size,
        paired_episodes=len(data),videos=len(media),zip_crc_passed=True,summaries=summaries)
    (ROOT/'adjust-three-seed-bundle.json').write_text(json.dumps(record,indent=2)+'\n')
    report=readme
    for x in media:report=report.replace(f"](videos/{x['cell']}/seed-{x['scene_seed']}.mp4)",f"]({x['video']})")
    report+=f'\n[Full-quality evidence bundle]({output})\n'
    for dest in [SOURCE/'ADJUST_THREE_SEED_RESULTS.md',ROOT/'ADJUST_THREE_SEED_RESULTS.md']:dest.write_text(report)
    print(json.dumps(record))

if __name__=='__main__':main()
