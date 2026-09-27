"""Fixed all-pair diagnostic analysis. Never select a model from these outcomes."""
import hashlib,json
from pathlib import Path
import numpy as np
from repeat_common import difference,payload_hash
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/repeatability-20260921'
def main():
    freeze=json.loads((OUT/'source-freeze.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in freeze['sources'].items())
    assert hashlib.sha256((OUT/'scenes.json').read_bytes()).hexdigest()==freeze['manifest_sha256']
    capture=json.loads((OUT/'capture-summary.json').read_text());inputs=sorted(x['name']for x in capture['records'])
    assert len(inputs)==6
    for row in capture['records']:
        assert row['actions_executed']==0
        p=json.loads((OUT/'inputs'/f"{row['name']}.json").read_text());assert payload_hash(p)==row['payload_sha256']
    records={};arrays={}
    for process,n in [('A',12),('B',42)]:
        d=OUT/f'process-{process}';complete=json.loads((d/'complete.json').read_text());assert complete['count']==n
        rows=json.loads((d/'records.json').read_text());assert len(rows)==n
        for row in rows:
            k=(process,row['arm'],row['rep'],row['input']);assert k not in records
            records[k]=row;arrays[k]=np.load(d/f"{row['name']}.npz")
    expected={('A','identity',rep,i)for rep in range(2)for i in inputs}|{('B','identity',0,i)for i in inputs}|{('B',arm,rep,i)for arm in ['renorm','adapter','adapter_renorm']for rep in range(2)for i in inputs}
    assert set(records)==expected
    comparisons=[]
    def compare(label,left,right):
        a,b=arrays[left],arrays[right]
        assert records[left]['payload_sha256']==records[right]['payload_sha256']
        metrics={field:difference(a[field],b[field])for field in ['chunk','first_action']}
        dims=records[left]['binary_command_dims'];assert dims==records[right]['binary_command_dims']
        if dims:metrics['first_action']['binary_command_disagreements']=int(np.count_nonzero((a['first_action'][dims]>.5)!=(b['first_action'][dims]>.5)))
        comparisons.append({'comparison':label,'input':left[-1],'candidate':list(left),'reference':list(right),**metrics})
    for i in inputs:
        compare('identity_within_A',('A','identity',1,i),('A','identity',0,i))
        for rep in range(2):compare(f'identity_restart_vs_A{rep}',('B','identity',0,i),('A','identity',rep,i))
        for arm in ['renorm','adapter','adapter_renorm']:
            compare(arm+'_within_B',('B',arm,1,i),('B',arm,0,i))
            for rep in range(2):compare(f'{arm}_r{rep}_vs_identity',('B',arm,rep,i),('B','identity',0,i))
    aggregates={}
    for label in sorted(set(x['comparison']for x in comparisons)):
        rows=[x for x in comparisons if x['comparison']==label]
        aggregates[label]={f:{'byte_equal_inputs':sum(r[f]['byte_equal']for r in rows),'inputs':len(rows),'max_abs':max(r[f]['max_abs']for r in rows),'mean_input_rmse':float(np.mean([r[f]['rmse']for r in rows]))}for f in ['chunk','first_action']}
    out={'role':'fixed-input repeatability and sensitivity diagnostic, not policy success','all_integrity_checks_passed':True,'generated_chunks':54,'unique_inputs':6,'scene_seeds':[400000,400001,400002],'aggregates':aggregates,'comparisons':comparisons,'limitations':['reused scenes','only initial observations','rot6d is not angular error','raw generated chunks differ from post-projection executed trajectories','no RAE versus VAE comparison']}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'generated_chunks':54,'aggregates':aggregates},indent=2))
if __name__=='__main__':main()
