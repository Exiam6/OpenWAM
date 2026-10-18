"""Generate the completed pilot report only after result, media and visual gates."""
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
    summary=json.loads((ROOT/'pilot-summary.json').read_text());assert summary['complete']
    audit=json.loads((ROOT/'independent-result-audit.json').read_text());assert audit['verified_finished_episodes'] and audit['inspected']==96 and not audit['issues']
    visual=json.loads((ROOT/'pilot-visual-review.json').read_text());assert visual['selected_contactsheets_inspected']
    expected={(f'{v}-seed{s}',t,scene) for v in p['variants'] for s in p['seeds'] for t,scenes in p['selected'].items() for scene in scenes}
    assert len(expected)==90
    samples={(f'{v}-seed{s}',t,p['selected'][t][0]) for v in p['variants'] for s in p['seeds'] for t in p['selected']}
    assert samples<={(x['cell'],x['task'],x['scene_seed']) for x in visual['sheets']}
    for x in visual['sheets']:assert sha(x['path'])==x['sha256']
    baseline=json.loads((ROOT/'baseline-selected.json').read_text());media=[];episodes=[]
    for task in p['selected']:
        for seed in p['seeds']:
            records=json.loads((ROOT/f'paired-videos-{task}-seed{seed}.json').read_text())['records'];assert len(records)==10;media.extend(records)
    assert {(x['cell'],x['task'],x['scene_seed']) for x in media}==expected and len(media)==90
    for cell,task,scene in sorted(expected):
        path=ROOT/'evaluation'/cell/'fixed'/task/f'seed-{scene}'/'result.json';result=json.loads(path.read_text());assert result['status']=='complete'
        check=next(x for x in audit['rows'] if (x['cell'],x['condition'],x['task'],x['scene_seed'])==(cell,'fixed',task,scene))
        assert check['verified'] and check['paired_first_chunk_max_action_difference']==0 and check['initial_state_exact']
        assert all(x==0 for x in check['all_three_camera_mae'].values())
        video=next(x for x in media if (x['cell'],x['task'],x['scene_seed'])==(cell,task,scene))
        assert video['new_result_sha256']==sha(path) and video['full_decode_passed'] and video['video_sha256']==sha(video['video'])
        before=next(x for x in baseline if (x['route'],x['training_seed'],x['task'],x['scene_seed'])==(result['route'],result['training_seed'],task,scene))
        assert video['old_success']==before['success'] and video['fixed_success']==result['success']
        episodes.append(dict(cell=cell,task=task,scene_seed=scene,old=before,fixed=result,audit=check,media=video,result_sha256=sha(path),trace_sha256=sha(path.parent/'trace.jsonl')))
    hand={v:sum(x['fixed']['success'] for x in episodes if x['fixed']['route']==v and x['task']=='handover_block') for v in p['variants']}
    verdict='pilot_transfer_supported' if max(hand.values())>=2 else ('no_handover_success_on_pilot' if max(hand.values())==0 else 'undecided')
    assert verdict==summary['verdict']
    aggregates=[]
    for v in p['variants']:
        for task in p['selected']:
            rows=[x for x in episodes if x['fixed']['route']==v and x['task']==task];assert len(rows)==15
            old=sum(x['old']['success'] for x in rows);fixed=sum(x['fixed']['success'] for x in rows)
            aggregates.append(dict(variant=v,task=task,n=len(rows),old=old,fixed=fixed,gains=sum(x['fixed']['success'] and not x['old']['success'] for x in rows),losses=sum(x['old']['success'] and not x['fixed']['success'] for x in rows),adjust_regression_flag=task=='adjust_bottle' and (old-fixed)/len(rows)>=.2-1e-12))
    lines=['# Native-calibration learned-policy pilot','',f'Completed:90 corrected episodes and six reproducing old-path controls. Registered primary verdict: **{verdict}**.', '',
        'The question is whether restoring verified native planner calibration raises learned-policy handover success above zero. '
        'The intervention preserves the ALOHA filename marker when relocating numerically identical planner configurations. '
        'Checkpoints at step20000, native inference, prompts, RNG, first-five frozen scenes per task, cameras and task success rules are unchanged. '
        'This is a development pilot with three training seeds and five shared scene IDs per task, not 90 independent scenes or an unseen test.', '',
        'The preregistered primary criterion is at least two handover successes out of15 for either model. Zero for both leaves a policy-learning obstacle after the execution fix; one success is undecided. '
        'An adjust success drop of at least20 percentage points is flagged as regression. Place is exploratory. '
        'This pilot cannot rank RAE architectures or invalidate the separate successful expert-execution repair.', '',
        '| Model | Task | Historical old execution | Restored calibration | Gains | Losses | Regression flag |','|---|---|---:|---:|---:|---:|---|']
    for x in aggregates:lines.append(f"| {x['variant']} | {x['task']} | {x['old']/x['n']:.0%} ({x['old']}/{x['n']}) | {x['fixed']/x['n']:.0%} ({x['fixed']}/{x['n']}) | {x['gains']} | {x['losses']} | {x['adjust_regression_flag']} |")
    lines+=['','Next: follow the separately registered corrected factorial to compare the existing architectures under the same repaired executor. '
        'Keep execution recovery and learned-policy behavior as distinct claims. Do not change model selection, task limits or scene membership in response to these results.', '',
        '## Every training seed','', '| Model | Seed | Task | Old | Fixed |','|---|---:|---|---:|---:|']
    for row in summary['rows']:
        assert row['finished']==row['planned']==5 and row['infrastructure_failures']==0
        lines.append(f"| {row['variant']} | {row['seed']} | {row['task']} | {row['baseline_on_finished']/5:.0%} ({row['baseline_on_finished']}/5) | {row['successes']/5:.0%} ({row['successes']}/5) |")
    lines+=['','## Validation and limits','',
        'All96 episode results pass independent auditing; the90 corrected pairs have exact initial states, zero initial camera error in all three views and matching first native action chunks. '
        'All90 videos fully decode and match result hashes. The first registered scene contact sheet of every model/seed/task group was visually reviewed at sampled start/intermediate/final frames. '
        'This is not manual inspection of every video frame. Native wrist cameras can be occluded; no additional external whole-robot view was recorded.', '',
        'Videos pair historical old execution above and restored calibration below. Camera order is head, left wrist, right wrist. '
        'Clips align action index at12fps rather than physical time; shorter episodes hold their final frame. '
        'All failed outcomes and the earlier intermediate evidence bundles remain available. No repeated failed attempt is silently substituted.', '',
        '## All90 paired clips','', '| Model and seed | Task | Scene | Old | Fixed | Video |','|---|---|---:|---|---|---|']
    for x in media:lines.append(f"| {x['cell']} | {x['task']} | {x['scene_seed']} | {x['old_success']} | {x['fixed_success']} | [Play / download](videos/{x['cell']}/{x['task']}/seed-{x['scene_seed']}.mp4) |")
    lines+=['',f"Registration commit: `{p['commit']}`. Source model commit: `{p['model_commit']}`.",
        'The archive includes every paired clip, contact sheet, per-episode result and independent audit, registered provenance and the sampled visual review.']
    readme='\n'.join(lines)+'\n';out=ROOT/'complete-pilot-evidence.zip';assert not out.exists(),'Never replace frozen evidence'
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_STORED) as z:
        z.writestr('README.md',readme);z.writestr('data.json',json.dumps(dict(verdict=verdict,aggregates=aggregates,episodes=episodes),indent=2)+'\n')
        for name in ['registration.json','pilot-summary.json','independent-result-audit.json','pilot-visual-review.json']:z.write(ROOT/name,name)
        for name in ['PLAN.md','registry.json','report_pilot.py','make_pairs.py','audit_results.py']:z.write(SOURCE/name,'source/'+name)
        for x in media:
            video=Path(x['video']);prefix=f"videos/{x['cell']}/{x['task']}/seed-{x['scene_seed']}"
            for ext in ['.mp4','.jpg','.json']:z.write(video.with_suffix(ext),prefix+ext)
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    record=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),path=str(out),sha256=sha(out),bytes=out.stat().st_size,videos=90,paired_episodes=90,zip_crc_passed=True,verdict=verdict,aggregates=aggregates)
    (ROOT/'complete-pilot-bundle.json').write_text(json.dumps(record,indent=2)+'\n')
    report=readme
    for x in media:report=report.replace(f"](videos/{x['cell']}/{x['task']}/seed-{x['scene_seed']}.mp4)",f"]({x['video']})")
    report+=f'\n[Full-quality evidence bundle]({out})\n'
    for path in [SOURCE/'RESULTS.md',ROOT/'RESULTS.md']:path.write_text(report)
    print(json.dumps(record))

if __name__=='__main__':main()
