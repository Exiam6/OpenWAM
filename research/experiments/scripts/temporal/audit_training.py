"""Verify every matched continuation, full checkpoint and paired batch stream."""
import hashlib,json
from pathlib import Path
from run_owned import write
R=Path('/data02/zifanz4/openwam-experiments/temporal-20260926');S=Path('/home/zifanz4/openwam-experiments/studies/temporal-20260926')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def audit():
 result={'passed':False,'seeds':{}}
 for seed in [42,43,44]:
  arms={}
  for arm in ['mean','learned']:
   out=R/'training'/f'{arm}-seed{seed}';assert json.loads((out/'exit.json').read_text())['returncode']==0
   metrics=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()];assert len(metrics)==4000
   assert metrics[-1]['global_step']==4000 and metrics[-1]['opt_step']==2000,metrics[-1]
   stream=(out/'batch-stream.jsonl').read_text().splitlines();assert len(stream)==4000
   assert json.loads((out/'encoder-parity.json').read_text())['passed']
   ckpts=list((out/'output').glob('*/checkpoint_step_4000.safetensors'));assert len(ckpts)==1
   arms[arm]={'initial':json.loads((out/'initialization.json').read_text())['source_sha256'],'batch_stream_sha256':sha(out/'batch-stream.jsonl'),'checkpoint':str(ckpts[0]),'checkpoint_sha256':sha(ckpts[0])}
  assert arms['mean']['initial']==arms['learned']['initial'];assert arms['mean']['batch_stream_sha256']==arms['learned']['batch_stream_sha256'];result['seeds'][str(seed)]=arms
 result['passed']=True;write(S/'training-audit.json',result)
if __name__=='__main__':audit()
