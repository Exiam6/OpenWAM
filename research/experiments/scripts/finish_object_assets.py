#!/usr/bin/env python3
"""Resume pinned object assets using checked ranges and a full SHA256."""
import concurrent.futures,hashlib,json,time,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'benchmarks/RoboTwin/assets'
SIZE=3737778549;SHA='6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27'
REV='981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042'
URL=f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{REV}/objects.zip'
part=DEST/'objects.zip.part';dest=DEST/'objects.zip'
if not dest.exists():
 start=part.stat().st_size if part.exists() else 0
 chunks=[(a,min(a+64*1024**2,SIZE)-1) for a in range(start,SIZE,64*1024**2)]
 def get(bounds):
  a,b=bounds;path=DEST/f'objects.range-{a}-{b}'
  if path.exists() and path.stat().st_size==b-a+1:return path
  for attempt in range(4):
   try:
    req=urllib.request.Request(URL+f'?range={a}-{b}&attempt={attempt}',headers={'Range':f'bytes={a}-{b}'})
    with urllib.request.urlopen(req,timeout=90) as response,path.open('wb') as f:
     assert response.status==206 and response.headers['Content-Range']==f'bytes {a}-{b}/{SIZE}'
     while block:=response.read(4*1024**2):f.write(block)
    assert path.stat().st_size==b-a+1
    print('RANGE',a,b,flush=True);return path
   except Exception:
    if attempt==3:raise
    time.sleep(2**attempt)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(get,chunks))
 with part.open('ab') as f:
  for path in paths:
   with path.open('rb') as src:
    while block:=src.read(4*1024**2):f.write(block)
 assert part.stat().st_size==SIZE
 h=hashlib.sha256()
 with part.open('rb') as f:
  while block:=f.read(8*1024**2):h.update(block)
 assert h.hexdigest()==SHA
 part.replace(dest)
 for path in paths:path.unlink()
for name,expected in [('objects.zip',SHA),('embodiments.zip','6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6')]:
 h=hashlib.sha256()
 with (DEST/name).open('rb') as f:
  while block:=f.read(8*1024**2):h.update(block)
 assert h.hexdigest()==expected
 with zipfile.ZipFile(DEST/name) as z:
  for item in z.infolist():assert (DEST/item.filename).resolve().is_relative_to(DEST.resolve())
  z.extractall(DEST)
 print('VERIFIED AND EXTRACTED',name,flush=True)
(DEST/'download-provenance.json').write_text(json.dumps({'revision':REV,'objects_sha256':SHA,'embodiments_sha256':'6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6','range_response_validated':True,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},indent=2))
