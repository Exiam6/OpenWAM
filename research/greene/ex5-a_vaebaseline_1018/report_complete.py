"""Complete matched VAE evidence, with parent cost accounting and real visual review."""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

from build_cache import sha
from full_media import ROOT,G,COMPARISONS
from render_index import render_html

SOURCE=Path(__file__).parent


def add_asset(assets,path,name):
    path=Path(path);assert path.is_file()
    if name in assets:
        assert assets[name]==path or sha(assets[name])==sha(path),'Archive path has conflicting bytes: '+name
    else:assets[name]=path
    return name


def gate_evidence(root,seeds):
    """Small evidence behind release decisions; model/cache binaries stay external."""
    root=Path(root);paths=set()
    def include(path,digest=None):
        path=Path(path);assert path.is_file(),path
        if digest is not None:assert sha(path)==digest,path
        paths.add(path);return json.loads(path.read_text()) if path.suffix=='.json' else None
    release=include(root/'full-training-release.json')
    for path,digest in release['evidence_sha256'].items():include(path,digest)
    contract=include(root/'deployment-contract/result.json')
    for row in contract['rows']:include(row['trace'],row['trace_sha256'])
    for seed in seeds:
        release=include(root/f'evaluation-release-seed{seed}.json')
        include(root/f'evaluation-budget-seed{seed}.json')
        include(release['visual_review'],release['visual_review_sha256'])
        media=include(root/f'pilot-media-seed{seed}.json',release['media_sha256'])
        for record in media['records']:include(Path(record['video']).with_suffix('.json'))
        for phase in ['training','pilot_evaluation','full_evaluation']:
            claim_path=root/'phase-claims'/f'{phase}-seed{seed}.json'
            claim=include(claim_path,release['parent_claim_sha256'] if phase=='pilot_evaluation' else None)
            result=include(claim['result_path'],claim['result_sha256'])
            if phase=='training':continue
            directory=Path(claim['result_path']).parent
            include(directory.parent/'reference-actions.json',result['reference_sha256'])
            include(directory/'smoke/result.json')
            for group in result['groups']:include(group['audit'])
    return sorted(paths)


def readiness():
    for script in [G/'code/readout.py',ROOT/'code/readout.py']:subprocess.run([sys.executable,str(script)],check=True,capture_output=True,text=True)
    summary=json.loads((ROOT/'success-summary.json').read_text());assert summary['complete'] and all(x==180 for x in summary['verified_by_variant'].values()),'Complete matched180-evaluation endpoints required'
    p=json.loads((ROOT/'gpu-registration.json').read_text());rows={tuple(x['identity']):x for x in summary['rows']};assert len(rows)==540
    release=json.loads((ROOT/'full-training-release.json').read_text());assert release['passed']
    for path,digest in release['evidence_sha256'].items():assert sha(path)==digest,path
    preflight=json.loads((ROOT/'preflight-run.json').read_text());assert preflight['passed'] and preflight['status']=='complete' and 0<preflight['occupied_seconds']<=9000
    occupied=preflight['occupied_seconds'];costs=[];claims=[]
    for seed in p['seeds']:
        cost=dict(seed=seed)
        for phase in ['training','pilot_evaluation','full_evaluation']:
            path=ROOT/'phase-claims'/f'{phase}-seed{seed}.json';claim=json.loads(path.read_text());assert claim['complete'] and claim['exit_code']==0
            assert claim['seed']==seed
            expected_result=ROOT/'runs'/f'vae48_native-seed{seed}'/'result.json' if phase=='training' else ROOT/'evaluation'/f'seed{seed}'/('pilot' if phase=='pilot_evaluation' else 'full')/'result.json'
            assert Path(claim['result_path'])==expected_result
            seconds=claim['occupied_seconds'];assert math.isfinite(seconds) and seconds>0
            assert sha(claim['result_path'])==claim['result_sha256'];cost[phase]=seconds;occupied+=seconds;claims.append(path)
        assert cost['training']<=10800 and cost['pilot_evaluation']<=3600 and cost['pilot_evaluation']+cost['full_evaluation']<=10800
        costs.append(cost)
    assert occupied<=p['max_occupied_seconds']==73800
    manifest=json.loads((ROOT/'full-media/manifest.json').read_text());assert manifest['complete'] and manifest['paired_videos']==360
    records={x['key']:x for x in manifest['records']};expected=set();sampled=set()
    for comparison in COMPARISONS:
        for seed in p['seeds']:
            for task,scenes in p['scenes'].items():
                for scene in scenes:
                    key=f'{comparison}/seed{seed}/{task}/scene-{scene}';expected.add(key)
                    if scene==scenes[0]:sampled.add(key)
    assert len(records)==len(manifest['records'])==360 and set(records)==expected
    for key,record in records.items():
        baseline,labels=COMPARISONS[record['comparison']];assert record['labels']==labels
        for side,variant in [('left',baseline),('right','vae48_native')]:
            identity=(variant,record['seed'],record['task'],record['scene_seed']);row=rows[identity];item=record[side]
            assert item['result_path']==row['result_path'] and item['result']==row['result']
            assert sha(item['result_path'])==item['result_sha256']==row['result_sha256']
            assert sha(Path(item['result_path']).parent/'trace.jsonl')==item['trace_sha256']==row['trace_sha256']
            assert record[f'{side}_success']==item['result']['success']
        assert record['full_decode_passed'] and sha(record['video'])==record['video_sha256'] and sha(record['preview'])==record['preview_sha256']
    review_path=ROOT/'full-media-visual-review.json';review=json.loads(review_path.read_text())
    assert review['complete'] and review['visually_inspected'] and review['scope']=='six sampled action indices on the first scene of every comparison/task/training-seed cell'
    assert len(review['records'])==18 and {x['key'] for x in review['records']}==sampled
    for item in review['records']:
        record=records[item['key']];assert item['preview_sha256']==record['preview_sha256'] and item['indices']==record['preview_indices'] and item['observations'].strip()
    gate_evidence(ROOT,p['seeds'])
    return p,summary,records,occupied,costs,claims,review_path


def training_figure(destination,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for seed in [42,43,44]:
        rows=[json.loads(x) for x in (ROOT/'runs'/f'vae48_native-seed{seed}'/'metrics.jsonl').read_text().splitlines()]
        assert rows[-1]['step']==20000 and all(b['step']>a['step'] for a,b in zip(rows,rows[1:]))
        for ax,key in zip(axes.flat,['loss','loss_action','loss_video']):
            assert all(math.isfinite(x[key]) for x in rows)
            ax.plot([x['step'] for x in rows],[x[key] for x in rows],label=f'Seed {seed}',alpha=.8,linewidth=.8);ax.set(xlabel='Training step',ylabel=key);ax.grid(alpha=.2);ax.legend()
    first=next(x for x in records.values() if x['comparison']=='primary_raw_minus_vae' and x['seed']==42 and x['task']=='adjust_bottle' and x['scene_seed']==6200001)
    axes[1,1].imshow(plt.imread(first['preview']));axes[1,1].axis('off');axes[1,1].set_title('Raw / VAE: first fixed adjustment scene')
    fig.suptitle('Matched VAE training losses and paired rollout sample')
    fig.text(.5,.015,'Losses describe VAE training only, not closed-loop success or cross-representation equivalence. Fixed endpoint: 20000 steps.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.96]);path=destination/'training-and-rollout.png';fig.savefig(path,dpi=160);plt.close(fig);return path


def generate():
    p,summary,records,occupied,costs,claims,review=readiness();destination=ROOT/'full-report';bundle=ROOT/'complete-vae-evidence.zip'
    assert not destination.exists() and not bundle.exists(),'Preserve previous report attempt'
    assets={};entries=[]
    def add(path,name):return add_asset(assets,path,name)
    for key,record in sorted(records.items()):
        paths={field:add(record[field],f'{field}s/{key}'+Path(record[field]).suffix) for field in ['video','preview']}
        metadata=add(ROOT/'full-media'/f'{key}.json',f'pairs/{key}.json')
        a,b=record['left_success'],record['right_success'];outcome='Both succeed' if a and b else ('Right-only success' if b else ('Left-only success' if a else 'Both fail'))
        entries.append(dict(key=key,comparison=record['comparison'],task=record['task'],seed=record['seed'],scene=record['scene_seed'],labels=record['labels'],outcome=outcome,metadata=metadata,**paths))
        for side in ['left','right']:
            path=Path(record[side]['result_path'])
            for item in [path,path.parent/'trace.jsonl']:add(item,'evidence/'+str(item.relative_to(ROOT.parent)))
    for path in [ROOT/'gpu-registration.json',ROOT/'gpu-registration-audit.json',ROOT/'success-summary.json',ROOT/'full-training-release.json',ROOT/'preflight-run.json',ROOT/'full-media/manifest.json',review,*claims]:add(path,'evidence/'+str(path.relative_to(ROOT.parent)))
    gate_index=[]
    for path in gate_evidence(ROOT,p['seeds']):
        name=add(path,'evidence/'+str(path.relative_to(ROOT.parent)))
        gate_index.append(dict(source_path=str(path),archive_path=name,sha256=sha(path)))
    for seed in p['seeds']:
        run=ROOT/'runs'/f'vae48_native-seed{seed}'
        for name in ['metrics.jsonl','result.json','initial-dev.json','final-dev.json','batches.npy','config.yaml']:add(run/name,f'training/seed{seed}/{name}')
    for name,digest in p['files'].items():assert sha(name)==digest;add(name,f'frozen/{digest}-{Path(name).name}')
    for name in ['report_complete.py','full_media.py','paired_video.py','render_index.py','FULL_EVIDENCE.md']:add(SOURCE/name,'source/'+name)
    primary=summary['comparisons']['primary_raw_minus_vae']
    verdict={'useful_raw_dino_benefit_supported':'The registered useful Raw-DINO benefit is supported on this development cohort.','positive_benefit_refuted_on_this_cohort':'A positive Raw-DINO benefit is refuted on this development cohort.','undecided':'The registered Raw-DINO benefit remains undecided on this development cohort.'}[primary['verdict']]
    body='# Matched DINO / Wan VAE: complete evidence\n\n'+verdict+'\n\n'
    body+='Question: does Raw DINO improve native closed-loop success over Wan VAE under matched data, initialization where shapes agree, minibatches, training steps and calibrated evaluation? The intervention changes the frozen visual encoder and latent channels. The12-layer1024-wide video core,512-wide action model,20Dstate/action, native heads/shifts5, batch3with accumulation2, seeds42/43/44 and fixed20000-step endpoint are held. Dimension-specific projections and frozen pretraining necessarily differ. The official larger policy is not included.\n\n'
    body+='Criterion registered before GPU work: Raw-minus-VAE at least5percentage points, positive in at least two training seeds, with positive lower95%paired bootstrap bound. A nonpositive upper bound refutes positive benefit; otherwise undecided.10000crossed seed/within-task-scene draws use RNG921018.\n\n'
    body+=f"Primary difference: **{primary['difference']*100:+.2f} pp**,95%interval **[{primary['ci95'][0]*100:+.2f}, {primary['ci95'][1]*100:+.2f}] pp**. Per-seed differences: "+', '.join(f'{x*100:+.2f}pp' for x in primary['per_seed_difference'])+'.\n\n'
    body+='| Corrected model | Adjust | Handover | Placement | Overall |\n|---|---|---|---|---|\n'
    for variant in ['raw768_native','svae48_native','vae48_native']:
        rows=[x for x in summary['rows'] if x['identity'][0]==variant];values=[]
        for task in p['tasks']:
            selected=[x for x in rows if x['identity'][2]==task];n=sum(x['result']['success'] for x in selected);values.append(f'{n/60:.1%} ({n}/60)')
        n=sum(x['result']['success'] for x in rows);body+='| '+variant+' | '+' | '.join(values+[f'{n/180:.1%} ({n}/180)'])+' |\n'
    body+='\n| Model | Seed42 | Seed43 | Seed44 |\n|---|---|---|---|\n'
    for variant in ['raw768_native','svae48_native','vae48_native']:
        values=[]
        for seed in p['seeds']:
            n=sum(x['result']['success'] for x in summary['rows'] if x['identity'][:2]==[variant,seed]);values.append(f'{n/60:.1%} ({n}/60)')
        body+='| '+variant+' | '+' | '.join(values)+' |\n'
    secondary=summary['comparisons']['secondary_compact_minus_vae'];body+=f"\nSecondary Compact-minus-VAE difference: {secondary['difference']*100:+.2f}pp,95%interval [{secondary['ci95'][0]*100:+.2f}, {secondary['ci95'][1]*100:+.2f}]pp. This descriptive control does not replace the primary endpoint.\n\n"
    body+=f'New study occupation: **{occupied/3600:.3f} GPU-hours**, within the20.5-hour cap, including preflight, training and evaluation parent lifetimes. Baseline training/evaluation is historical and not charged again. Equal data/steps do not imply equal FLOPs or wall time.\n\n| VAE seed | Training GPUh | Pilot+remaining evaluation GPUh |\n|---|---|---|\n'
    for cost in costs:body+=f"| {cost['seed']} | {cost['training']/3600:.3f} | {(cost['pilot_evaluation']+cost['full_evaluation'])/3600:.3f} |\n"
    body+='\n![Training losses and paired rollout sample](training-and-rollout.png)\n\n[Download figure](training-and-rollout.png) · [Interactive videos](index.html) · [Download index](index.json) · [Gate evidence and hashes](gate-evidence-index.json)\n\n'
    body+='All360paired videos were fully decoded/hash checked. Actual visual review samples six indices on each comparison/task/seed first scene (18sheets), not every frame. The baseline is on top, VAE below; numerical effects are baseline minus VAE. All three policy cameras are shown, aligned by action index with final-frame hold. No external whole-robot view was captured. Wrist occlusion and cropped objects remain visible.\n\n'
    body+='The180evaluations per arm cross60distinct development scenes with three training seeds. These known scenes cannot establish unseen-benchmark generalization; three seeds cannot establish broad population equivalence. Encoder pretraining and representation are not independently identified. Policy failure remains an outcome, and no checkpoint or scene was selected by success.\n\n| Comparison | Task | Seed | Scene | Outcome | Video |\n|---|---|---|---|---|---|\n'
    for entry in entries:body+=f"| {entry['comparison']} | {entry['task']} | {entry['seed']} | {entry['scene']} | {entry['outcome']} | [Download]({entry['video']}) |\n"
    destination.mkdir()
    for name,path in assets.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(path)
    figure=training_figure(destination,records)
    (destination/'gate-evidence-index.json').write_text(json.dumps(gate_index,indent=2)+'\n')
    (destination/'RESULTS.md').write_text(body);(destination/'index.json').write_text(json.dumps(entries,indent=2));page=render_html(entries,summary,True).replace('<header>','<header><p><a href="training-and-rollout.png" download>Download training figure</a></p>');(destination/'index.html').write_text(page)
    for entry in entries:
        for field in ['video','preview','metadata']:assert (destination/entry[field]).is_file()
    with zipfile.ZipFile(bundle,'x') as z:
        for name,path in sorted(assets.items()):z.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix in ['.mp4','.jpg'] else zipfile.ZIP_DEFLATED,compresslevel=1)
        z.write(figure,figure.name);z.writestr('README.md',body);z.writestr('RESULTS.md',body);z.writestr('index.json',json.dumps(entries,indent=2));z.writestr('index.html',render_html(entries,summary,False))
        z.writestr('gate-evidence-index.json',json.dumps(gate_index,indent=2))
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None;names=set(z.namelist());assert all(entry[field] in names for entry in entries for field in ['video','preview','metadata'])
        assert all(item['archive_path'] in names for item in gate_index)
    result=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),complete=True,evaluations_per_arm=180,paired_videos=360,visually_reviewed_sheets=18,occupied_gpu_hours=occupied/3600,report=str(destination/'RESULTS.md'),page=str(destination/'index.html'),bundle=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),zip_crc_passed=True)
    with (ROOT/'complete-evidence.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-readiness',action='store_true');args=parser.parse_args()
    if args.check_readiness:
        try:readiness();print('ready')
        except AssertionError as error:print(str(error));raise SystemExit(2)
    else:generate()
