"""Read-only audit of retained matched fits; no predictions or optimization."""
import datetime,hashlib,json,os
from pathlib import Path
import torch
ROOT=Path(os.environ['WAM_ROOT']);O=ROOT/'results/goalaux-20260922'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 torch.set_num_threads(2)
 assert (O/'training-exit-code.txt').read_text().strip()=='0'
 complete=json.loads((O/'training-complete.json').read_text());assert len(complete['models'])==6
 protocol=json.loads((O/'matched-protocol.json').read_text());inputs=json.loads((O/'training-inputs.json').read_text());assert inputs['source_protocol_sha256']==sha(O/'matched-protocol.json')
 assert len(inputs['sha256'])==105 and len(inputs['rows'])==1260 and all(r['split']=='train' for r in inputs['rows'])
 for task in ['adjust_bottle','handover_block','place_object_basket']:
  rows=[r for r in inputs['rows'] if r['task']==task];eps=sorted(set(r['episode'] for r in rows));assert len(eps)==35 and len(rows)==420
  assert all(sum(r['episode']==ep for r in rows)==12 for ep in eps)
 for name,expected in protocol['source_sha256'].items():assert sha(ROOT/name)==expected,name
 pairs=[]
 for seed in [42,43,44]:
  states=[];records=[]
  for arm,weight in [('reconstruction_only',0.),('goal_auxiliary',0.1)]:
   path=O/'models'/arm/f'seed{seed}';rec=json.loads((path/'complete.json').read_text());assert rec in complete['models'];assert rec['seed']==seed and rec['arm']==arm and rec['steps']==2000
   assert sha(path/'final.pt')==rec['checkpoint_sha256'];assert sha(ROOT/f'results/layers-20260922/L12/seed{seed}/final.pt')==rec['initial_checkpoint_sha256']
   ck=torch.load(path/'final.pt',map_location='cpu',weights_only=False);assert ck['seed']==seed and ck['arm']==arm and ck['step']==2000 and ck['auxiliary_weight']==weight and ck['source']=='L12'
   assert ck['training_input_manifest_sha256']==sha(O/'training-inputs.json')
   assert all(torch.isfinite(t).all() for t in ck['state_dict'].values())
   states.append(ck);records.append(rec)
  assert records[0]['initial_checkpoint_sha256']==records[1]['initial_checkpoint_sha256']
  assert records[0]['sample_sequence_sha256']==records[1]['sample_sequence_sha256']
  assert states[0]['model_config']==states[1]['model_config']
  assert all(torch.equal(states[0]['state_dict'][n],states[1]['state_dict'][n]) for n in ['input_mean','input_std'])
  diffs={n:float((t-states[1]['state_dict'][n]).abs().max()) for n,t in states[0]['state_dict'].items() if n.startswith('enc_')}
  assert max(diffs.values())>0
  pairs.append({'seed':seed,'initial_checkpoint_identical':True,'sample_sequence_identical':True,'input_stats_identical':True,'all_final_tensors_finite':True,'encoder_max_abs_weight_difference':max(diffs.values()),'both_steps':2000,'model_config':states[0]['model_config']})
  del states
 result={'passed':True,'checked_at':datetime.datetime.now().astimezone().isoformat(),'fit_count':6,'paired_seeds':pairs,'training_episodes':105,'training_clips':1260,'all_source_and_final_checkpoint_hashes_match':True,'readouts_or_confirmation_scored':False,'policy_gain_claim':False,'scope':'Artifact/paired-design integrity only; representation benefit unmeasured'}
 (O/'training-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
