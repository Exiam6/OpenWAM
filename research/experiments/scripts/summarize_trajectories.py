"""Compare all six prespecified original-policy trajectory pairs."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
from repeat_common import array_hash,payload_hash
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT/'results/repeatability-20260921';OUT=PARENT/'closedloop'

def first_difference(a,b,key):
    return next((i for i,(x,y)in enumerate(zip(a,b))if x[key]!=y[key]),None)

def main():
    freeze=json.loads((OUT/'source-freeze.json').read_text())
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in freeze['sources'].items())
    assert hashlib.sha256((OUT/'protocol.md').read_bytes()).hexdigest()==freeze['protocol_sha256']
    manifest=json.loads((PARENT/'scenes.json').read_text());assert hashlib.sha256((PARENT/'scenes.json').read_bytes()).hexdigest()==freeze['manifest_sha256']
    seeds=[r['seed']for r in manifest['scenes']];assert seeds==[400000,400001,400002]
    initial={(r['condition'],r['seed']):r['payload_sha256']for r in json.loads((PARENT/'capture-summary.json').read_text())['records']}
    groups={};traces={};total_requests=0;payload_bytes=0
    for round_id in [0,1]:
        server=json.loads((OUT/f'round{round_id}-server-check.json').read_text());assert server['encoder_weights_exact'] and server['sync_executor'] and server['inference_horizon']is None
        for condition in ['clean','noise']:
            name=f'r{round_id}-{condition}';d=OUT/name
            assert (d/'exit-code.txt').read_text().strip()=='0'
            data=json.loads((d/'summary.json').read_text());assert data['episodes']==3 and [r['seed']for r in data['records']]==seeds
            vids=list((d/'runtime/eval_result').rglob('episode*.mp4'));assert len(vids)==3
            for row in data['records']:
                seed=row['seed'];records=[json.loads(line)for line in (d/f'seed-{seed}-trace.jsonl').read_text().splitlines()]
                assert len(records)==row['actions']==row['calls'] and 0<len(records)<=400
                assert records[0]['request_sha256']==initial[(condition,seed)]
                assert bool(records[-1]['success_after_action'])==row['success']
                for step,rec in enumerate(records):
                    assert rec['step']==step and rec['seed']==seed and rec['condition']==condition
                    assert rec['requests']==rec['actions']==1 and rec['server_step']==rec['after_action_count']==step+1
                    assert rec['action_type']=='ee'
                    for field,hash_field,dtype in [('server_action','server_action_hash','float32'),('env_action','env_action_hash',rec['env_action_dtype']),('eef_state','eef_hash','float32'),('joint_state','joint_hash',rec['joint_dtype'])]:
                        a=np.asarray(rec[field],dtype=dtype);assert np.isfinite(a).all()and array_hash(a)==rec[hash_field]
                    path=d/rec['request_file'];raw=path.read_bytes();payload_bytes+=len(raw)
                    payload=json.loads(gzip.decompress(raw));assert payload_hash(payload)==rec['request_sha256']
                traces[(round_id,condition,seed)]=records;total_requests+=len(records)
            groups[name]=data
    comparisons=[]
    for condition in ['clean','noise']:
        for seed in seeds:
            a=traces[0,condition,seed];b=traces[1,condition,seed];shared=min(len(a),len(b))
            fields={'raw_images':'raw_cameras','eef_state':'eef_hash','joint_state':'joint_hash','policy_images':'policy_cameras','request':'request_sha256','server_action':'server_action_hash','env_target':'env_action_hash','success_flag':'success_after_action'}
            first={label:first_difference(a,b,key)for label,key in fields.items()}
            cameras={camera:next((i for i,(x,y)in enumerate(zip(a,b))if x['raw_cameras'][camera]!=y['raw_cameras'][camera]),None)for camera in a[0]['raw_cameras']}
            diffs={field:float(np.max(np.abs(np.array([x[field]for x in a[:shared]],dtype=np.float64)-np.array([x[field]for x in b[:shared]],dtype=np.float64))))for field in ['eef_state','joint_state','server_action','env_action']}
            old=np.load(PARENT/f'process-B/identity-r0-{condition}-{seed}.npz')['first_action']
            initial_comparison=[{'round':i,'exact':np.array_equal(np.asarray(trace[0]['server_action'],dtype=np.float32),old),'max_abs':float(np.max(np.abs(np.asarray(trace[0]['server_action'],dtype=np.float32)-old)))}for i,trace in enumerate([a,b])]
            comparisons.append({'condition':condition,'seed':seed,'lengths':[len(a),len(b)],'successes':[a[-1]['success_after_action'],b[-1]['success_after_action']],'shared_steps':shared,'first_different_step':first,'raw_camera_first_differences':cameras,'max_abs_over_shared_prefix':diffs,'all_recorded_values_identical':len(a)==len(b)and all(v is None for v in first.values()),'first_action_vs_fixed_request_diagnostic':initial_comparison})
    result={'role':'12 reused-scene original-policy repeatability rollouts, not generalization benchmark','all_integrity_checks_passed':True,'episodes':12,'requests':total_requests,'compressed_request_bytes':payload_bytes,'groups':groups,'paired_trajectories':comparisons,'limits':['three reused scenes','two repeats','differences do not isolate a physics/rendering/IK cause','no adapted-policy rollouts in this stage','finite repeatability checks cannot prove global determinism']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'episodes':12,'requests':total_requests,'paired_trajectories':comparisons},indent=2))
if __name__=='__main__':main()
