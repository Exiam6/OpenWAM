"""Hash-verified matched VAE endpoint; incomplete cells never enter a comparison."""
import datetime as dt
import json
from pathlib import Path
import numpy as np

from build_cache import sha
from metrics import compare

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
G=ROOT.parent/'ex4-g_calibratedfactorial_1017'
VARIANTS=['raw768_native','svae48_native','vae48_native']


def checked_row(row,expected):
    identity=tuple(row['identity']);assert identity in expected and row['verified']
    path=Path(row['result_path']);result=json.loads(path.read_text())
    assert sha(path)==row['result_sha256'] and result==row['result']
    assert sha(path.parent/'trace.jsonl')==row['trace_sha256']
    assert [result['route'],result['training_seed'],result['task'],result['scene_seed']]==list(identity)
    assert result['status']=='complete' and type(result['success']) is bool
    return identity


def collect():
    p=json.loads((G/'registration.json').read_text());tasks=p['tasks'];seeds=p['seeds'];scenes=p['scenes']
    assert seeds==[42,43,44] and len(tasks)==3 and all(len(scenes[t])==20 for t in tasks)
    expected={(v,s,t,c) for v in VARIANTS for s in seeds for t in tasks for c in scenes[t]}
    rows={};issues=[];completed=[]
    baseline=json.loads((G/'success-summary.json').read_text())
    for row in baseline['rows']:
        if row['identity'][0] not in VARIANTS[:2]:continue
        identity=checked_row(row,expected);assert identity not in rows;rows[identity]=dict(row,source='corrected-G')
    registration=ROOT/'gpu-registration.json';registered=registration.exists()
    if registered:
        vp=json.loads(registration.read_text());assert vp['scenes']==scenes
        for seed in seeds:
            trained_path=ROOT/'runs'/f'vae48_native-seed{seed}'/'result.json'
            for stage in ['pilot','full']:
                directory=ROOT/'evaluation'/f'seed{seed}'/stage;result_path=directory/'result.json'
                if not result_path.exists():continue
                result=json.loads(result_path.read_text())
                if result['status']=='failed':issues.append(dict(seed=seed,stage=stage,error=result.get('error')))
                if result['status']=='complete':
                    claim_path=ROOT/'phase-claims'/f'{"pilot_evaluation" if stage=="pilot" else "full_evaluation"}-seed{seed}.json'
                    if claim_path.exists():
                        claim=json.loads(claim_path.read_text())
                        if claim.get('complete') and claim.get('exit_code')==0:
                            assert claim['result_sha256']==sha(result_path);completed.append((seed,stage))
                trained=json.loads(trained_path.read_text());assert trained['complete'] and trained['steps']==20000
                assert sha(trained['checkpoint'])==trained['checkpoint_sha256']==result['checkpoint_sha256']
                for task in tasks:
                    audit_path=directory/'episodes'/task/'independent-audit.json'
                    if not audit_path.exists():continue
                    audit=json.loads(audit_path.read_text());assert audit['passed']
                    allowed=scenes[task][:5] if stage=='pilot' else scenes[task][5:]
                    assert {tuple(x['identity']) for x in audit['rows']}=={('vae48_native',seed,task,c) for c in allowed}
                    assert len(audit['rows'])==len(allowed)
                    for row in audit['rows']:
                        identity=checked_row(row,expected);assert identity not in rows
                        assert Path(row['result_path']).resolve()==(directory/'episodes'/task/f'seed-{identity[3]}'/'result.json').resolve()
                        rows[identity]=dict(row,source=f'VAE-{stage}')
    for seed in seeds:
        for task in tasks:
            for scene in scenes[task]:
                matched=[rows[(v,seed,task,scene)]['result'] for v in VARIANTS if (v,seed,task,scene) in rows]
                for item in matched[1:]:
                    for field in ['initial_state','action_seed','step_limit','evaluation_instruction']:assert item[field]==matched[0][field]
    complete=registered and baseline['complete'] and set(rows)==expected and len(completed)==6 and not issues
    summaries=[]
    for variant in VARIANTS:
        for seed in seeds:
            for task in tasks:
                selected=[x['result'] for k,x in rows.items() if k[:3]==(variant,seed,task)]
                summaries.append(dict(variant=variant,seed=seed,task=task,verified=len(selected),planned=20,successes=sum(x['success'] for x in selected),complete=len(selected)==20))
    comparisons={}
    if complete:
        arrays={v:np.array([[[int(rows[(v,s,t,c)]['result']['success']) for c in scenes[t]] for t in tasks] for s in seeds]) for v in VARIANTS}
        comparisons['primary_raw_minus_vae']=compare(arrays['raw768_native'],arrays['vae48_native'])
        secondary=compare(arrays['svae48_native'],arrays['vae48_native']);secondary['verdict']='secondary_descriptive_only'
        comparisons['secondary_compact_minus_vae']=secondary
    return dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),complete=complete,gpu_registered=registered,
        verified_by_variant={v:sum(k[0]==v for k in rows) for v in VARIANTS},planned_per_variant=180,
        baseline_complete=baseline['complete'],completed_vae_stages=completed,issues=issues,summaries=summaries,comparisons=comparisons,
        rows=list(rows.values()),scope='Same corrected executor;180 evaluations per arm,60 distinct scene seeds crossed with3training seeds. Unfinished cells pending. No comparison until all endpoints and parent jobs complete; official larger policy is not included.')


def main():
    result=collect();path=ROOT/'success-summary.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
    print(json.dumps({k:result[k] for k in ['complete','gpu_registered','verified_by_variant','completed_vae_stages','issues','comparisons']}))


if __name__=='__main__':main()
