#!/usr/bin/env python3
"""Download and verify the complete, pinned published DINO/S-VAE Study policy."""
import hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'benchmarks/RoboTwin/assets'
MODEL='TianxingChen/RoboTwin2.0'
REV='981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042'
DEST.mkdir(parents=True,exist_ok=True)
meta=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{MODEL}/tree/{REV}',timeout=30))
records=[]
for item in meta:
 if item['type']!='file' or item['path'] not in ['embodiments.zip','objects.zip']:continue
 rel=item['path'];p=DEST/rel;p.parent.mkdir(parents=True,exist_ok=True)
 assert p.resolve().is_relative_to(DEST.resolve())
 size=item['size'];url=f'https://huggingface.co/datasets/{MODEL}/resolve/{REV}/{rel}'
 if not p.exists():
  tmp=p.with_suffix(p.suffix+'.part')
  for attempt in range(5):
   try:
    offset=tmp.stat().st_size if tmp.exists() else 0
    headers={'Range':f'bytes={offset}-'} if offset else {}
    with urllib.request.urlopen(urllib.request.Request(url+f'?download=1&retry={attempt}',headers=headers),timeout=90) as response:
     if offset:assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {offset}-')
     else:assert response.status==200
     with tmp.open('ab' if offset else 'wb') as f:
      while block:=response.read(8*1024*1024):f.write(block)
    assert tmp.stat().st_size==size
    tmp.replace(p);break
   except Exception as e:
    print('RETRY',rel,attempt,type(e).__name__,flush=True)
    if attempt==4:raise
    time.sleep(2**attempt)
 assert p.stat().st_size==size
 h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(8*1024*1024):h.update(block)
 if 'lfs' in item:assert h.hexdigest()==item['lfs']['oid']
 records.append({'path':rel,'bytes':size,'sha256':h.hexdigest(),'lfs_hash_verified':'lfs' in item})
 print('VERIFIED',rel,size,flush=True)
(DEST/'download-provenance.json').write_text(json.dumps({'model':MODEL,'revision':REV,'files':records,'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},indent=2))
print('ROBOTWIN CLEAN ASSETS COMPLETE',flush=True)

import zipfile
for name in ['embodiments.zip','objects.zip']:
 with zipfile.ZipFile(DEST/name) as z:
  for item in z.infolist():assert (DEST/item.filename).resolve().is_relative_to(DEST.resolve())
  z.extractall(DEST)
 print('EXTRACTED',name,flush=True)
