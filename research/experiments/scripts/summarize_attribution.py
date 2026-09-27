"""Audit and compare every fixed-command planning/physics trace."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
from repeat_common import array_hash,payload_hash
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/attribution-20260921';PARENT=ROOT/'results/repeatability-20260921'

def validate_packed(x):
    if isinstance(x,dict):
        if set(['dtype','shape','values','sha256'])<=set(x):
            a=np.asarray(x['values'],dtype=x['dtype']);assert list(a.shape)==x['shape']and array_hash(a)==x['sha256']and np.isfinite(a).all()
        else:
            for v in x.values():validate_packed(v)
    elif isinstance(x,list):
        for v in x:validate_packed(v)

def diff_array(a,b):
    assert a['dtype']==b['dtype']and a['shape']==b['shape']
    x=np.array(a['values'],dtype=np.float64);y=np.array(b['values'],dtype=np.float64)
    return {'exact':a['sha256']==b['sha256'],'max_abs':float(np.abs(x-y).max())if x.size else 0.}

def main():
    freeze=json.loads((OUT/'source-freeze.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(sha(ROOT/k)==v for k,v in freeze['sources'].items())
    for name,field in [('protocol.md','protocol_sha256'),('fixed-commands.json','commands_sha256'),('scenes.json','manifest_sha256')]:assert sha(OUT/name)==freeze[field]
    assert sha(PARENT/'capture-summary.json')==freeze['capture_sha256']and sha(PARENT/'closedloop/summary.json')==freeze['stageC_summary_sha256']
    commands=json.loads((OUT/'fixed-commands.json').read_text())['commands'];scenes=json.loads((OUT/'scenes.json').read_text())['scenes'];seeds=[x['seed']for x in scenes];assert seeds==[400000,400001,400002]
    initial={x['seed']:x['payload_sha256']for x in json.loads((PARENT/'capture-summary.json').read_text())['records']if x['condition']=='clean'}
    traces={};total=0;video_count=0
    for rid in[0,1]:
        d=OUT/f'round{rid}';assert(d/'exit-code.txt').read_text().strip()=='0'
        s=json.loads((d/'summary.json').read_text());assert s['policy_inference_calls']==0 and s['step_limit_override']==2 and s['episodes']==3
        assert [r['seed']for r in s['records']]==seeds and all(r['calls']==r['actions']==2 for r in s['records'])
        vids=list((d/'runtime/eval_result').rglob('episode*.mp4'));assert len(vids)==3;video_count+=len(vids)
        for seed in seeds:
            rows=[json.loads(x)for x in(d/f'seed-{seed}-trace.jsonl').read_text().splitlines()];assert len(rows)==2
            assert rows[0]['request_sha256']==initial[seed]
            for step,row in enumerate(rows):
                assert row['seed']==seed and row['step']==step and row['condition']=='clean'
                assert row['requests']==row['actions']==1 and row['server_step']==row['after_action_count']==step+1
                command=commands[str(seed)][step]
                assert array_hash(np.asarray(row['server_action'],np.float32))==row['server_action_hash']==command['server_action_hash']
                assert array_hash(np.asarray(row['env_action'],dtype=row['env_action_dtype']))==row['env_action_hash']==command['env_action_hash']
                payload=json.loads(gzip.decompress((d/row['request_file']).read_bytes()));assert payload_hash(payload)==row['request_sha256']
                validate_packed(row)
                for name in ['before_execution','after_execution']:
                    sn=row[name];assert not sn['communication_flag'];assert payload_hash({k:v for k,v in sn.items()if k!='sha256'})==sn['sha256']
                assert [p['side']for p in row['planner_calls']]==['left','right']
                for p in row['planner_calls']:assert payload_hash(p['inputs'])==p['inputs_sha256']and payload_hash(p['result'])==p['result_sha256']
                total+=1
            traces[rid,seed]=rows
    comparisons=[]
    for seed in seeds:
        steps=[]
        for step,(a,b)in enumerate(zip(traces[0,seed],traces[1,seed])):
            states={}
            for name in ['before_execution','after_execution']:
                x,y=a[name],b[name]
                states[name]={'all_recorded_exact':x['sha256']==y['sha256'],'actual_articulations_exact':x['actual_articulations']==y['actual_articulations'],'actual_values':{side:{field:diff_array(x['actual_articulations'][side][field],y['actual_articulations'][side][field])for field in['qpos','qvel']}for side in['left','right']},'ee_link_poses_exact':x['ee_link_poses']==y['ee_link_poses'],'object_poses_exact':x['object_poses']==y['object_poses'],'drive_targets':diff_array(x['joint_drive_targets'],y['joint_drive_targets'])}
            plans=[]
            for x,y in zip(a['planner_calls'],b['planner_calls']):
                result={'side':x['side'],'inputs_exact':x['inputs_sha256']==y['inputs_sha256'],'results_exact':x['result_sha256']==y['result_sha256'],'statuses':[x['result']['status'],y['result']['status']]}
                if all(z=='Success'for z in result['statuses']):
                    for field in['position','velocity']:
                        result[field]=diff_array(x['result'][field],y['result'][field])if x['result'][field]['shape']==y['result'][field]['shape']else {'exact':False,'shapes':[x['result'][field]['shape'],y['result'][field]['shape']]}
                plans.append(result)
            old=[json.loads(x)for x in(PARENT/f'closedloop/r0-clean/seed-{seed}-trace.jsonl').read_text().splitlines()][step]
            steps.append({'step':step,'request_exact':a['request_sha256']==b['request_sha256'],'raw_cameras_exact':a['raw_cameras']==b['raw_cameras'],'eef_exact':a['eef_hash']==b['eef_hash'],'fixed_commands_exact':a['server_action_hash']==b['server_action_hash']and a['env_action_hash']==b['env_action_hash'],'states':states,'planners':plans,'request_matches_stageC_round0':[r['request_sha256']==old['request_sha256']for r in[a,b]]})
        comparisons.append({'seed':seed,'steps':steps})
    assert total==12 and video_count==6
    result={'role':'six two-command open-loop traces, not policy success evaluation','all_integrity_checks_passed':True,'traces':6,'commands_executed':total,'policy_inference_calls':0,'video_count':video_count,'comparisons':comparisons,'limits':['three reused clean scenes','two repetitions and two commands only','later scenes have truncated preceding-command context','measured state does not include bottle velocities or hidden solver state','ordering does not establish a specific library defect']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'traces':6,'commands':total,'comparisons':comparisons},indent=2))
if __name__=='__main__':main()
