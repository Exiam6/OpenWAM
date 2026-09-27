import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/norm-20260921'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reference=OUT/'reference-clean/summary.json'
s=json.loads(reference.read_text());rows=s['records']
assert len(rows)==10 and len({r['seed'] for r in rows})==10
assert all(400000<=r['seed']<500000 and r['condition']=='clean' for r in rows)
assert not (OUT/'fresh-scenes.json').exists()
keys=['seed','instruction','initial_head_sha256','initial_state_sha256']
manifest={'selection_rule':'First ten expert-feasible reference starts from 400000, independent of policy outcome; reference outcomes excluded from primary analysis','reference_summary_sha256':sha(reference),'scenes':[{**{k:r[k] for k in keys},'instruction_choice_length':10} for r in rows]}
(OUT/'fresh-scenes.json').write_text(json.dumps(manifest,indent=2))
print('FROZEN', [r['seed'] for r in rows],sha(OUT/'fresh-scenes.json'))
