#!/usr/bin/env python3
"""Two revision-pinned public task archives, streamed and SHA256 verified."""
import hashlib,json,urllib.request,zipfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'data'
REV='981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042'
SPECS={'handover_block':(546033461,'7e9e7145c3f1a4093d7a24a44dc5c0b1af2ccc551c3bc8e50d7bdde824da4947'),'place_object_basket':(635212427,'76518a717756589fc6787741e83dff24c6467e4506de8f7c5994008765f094d2')}
for task,(size,sha) in SPECS.items():
 dest=ROOT/(task+'_clean50.zip');url=f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{REV}/dataset/{task}/aloha-agilex_clean_50.zip'
 if not dest.exists():
  for attempt in range(3):
   try:
    tmp=dest.with_suffix('.part');h=hashlib.sha256();n=0
    with urllib.request.urlopen(url,timeout=90) as src,tmp.open('wb') as out:
     while block:=src.read(4*1024*1024):out.write(block);h.update(block);n+=len(block)
    assert n==size and h.hexdigest()==sha,(n,h.hexdigest());tmp.replace(dest);break
   except Exception:
    if attempt==2:raise
    time.sleep(2**attempt)
 h=hashlib.sha256()
 with dest.open('rb') as f:
  while block:=f.read(4*1024*1024):h.update(block)
 assert h.hexdigest()==sha and dest.stat().st_size==size
 target=ROOT/task;target.mkdir(exist_ok=True)
 with zipfile.ZipFile(dest) as z:
  for member in z.infolist():assert (target/member.filename).resolve().is_relative_to(target.resolve())
  if len(list(target.rglob('*.hdf5')))!=50:z.extractall(target)
 assert len(list(target.rglob('*.hdf5')))==50
 (target/'download-provenance.json').write_text(json.dumps({'task':task,'revision':REV,'sha256':sha,'bytes':size,'url':url},indent=2))
 print('VERIFIED',task,size,sha,flush=True)
