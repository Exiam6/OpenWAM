#!/usr/bin/env python3
"""Download public, revision-pinned pilot assets; extract only needed tensor ranges."""
import json,struct,urllib.request,hashlib,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL='OpenWAM/robotwin_dual_system_joint_self_attention_dinov3_svae'
REV='af1595c8ee54955116abbc0b0ffa17fa48deb691'
DATAREV='981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042'
BASE=f'https://huggingface.co/{MODEL}/resolve/{REV}/'
ASSET=ROOT/'assets/dinov3-study';ASSET.mkdir(parents=True,exist_ok=True)
def get(url,start=None,end=None):
 for attempt in range(4):
  try:
   headers={} if start is None else {'Range':f'bytes={start}-{end}'}
   req=urllib.request.Request(url,headers=headers)
   with urllib.request.urlopen(req,timeout=120) as r:
    if start is not None:
     assert r.status==206,(r.status,url)
     assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),r.headers['Content-Range']
    b=r.read()
    if start is not None:assert len(b)==end-start+1
    return b
  except Exception:
   if attempt==3:raise
   time.sleep(2**attempt)
for f in ['config.yaml','svae_config.json','dinov3/config.json','README.md']:
 p=ASSET/f;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.write_bytes(get(BASE+f))
url=BASE+'checkpoint_step_9890.safetensors'
headlen=struct.unpack('<Q',get(url+'?pilot=length',0,7))[0]
header=json.loads(get(url+'?pilot=header',8,7+headlen))
items=sorted(((k,v) for k,v in header.items() if k.startswith('video_backbone.video_encoder.')),key=lambda kv:kv[1]['data_offsets'][0])
assert len(items)>100
ranges=[]
for k,v in items:
 a,b=v['data_offsets']
 if ranges and a==ranges[-1][1]:ranges[-1][1]=b
 else:ranges.append([a,b])
print('Selected',len(items),'tensors,',sum(b-a for a,b in ranges),'bytes in',len(ranges),'ranges',flush=True)
segments=[]
for i,(a,b) in enumerate(ranges):
 p=ASSET/f'encoder-range-{i}.bin'
 if not p.exists() or p.stat().st_size!=b-a:
  p.write_bytes(get(url+f'?pilot=range{i}',8+headlen+a,8+headlen+b-1))
 segments.append((a,b,p.read_bytes()))
newh={};raw=[];offset=0
for k,v in items:
 a,b=v['data_offsets'];part=next(blob[a-lo:b-lo]for lo,hi,blob in segments if lo<=a and hi>=b)
 newh[k]={'dtype':v['dtype'],'shape':v['shape'],'data_offsets':[offset,offset+len(part)]};offset+=len(part);raw.append(part)
h=json.dumps(newh,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
out=ASSET/'encoder.safetensors'
with out.open('wb')as f:
 f.write(struct.pack('<Q',len(h)));f.write(h)
 for p in raw:f.write(p)
for i in range(len(ranges)):(ASSET/f'encoder-range-{i}.bin').unlink()
provenance={'model_repo':MODEL,'model_revision':REV,'source_checkpoint':'checkpoint_step_9890.safetensors','tensor_count':len(items),'tensor_bytes':offset,'extracted_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'data_repo':'TianxingChen/RoboTwin2.0','data_revision':DATAREV,'data_path':'dataset/adjust_bottle/aloha-agilex_clean_50.zip','data_expected_sha256':'5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e'}
(ASSET/'provenance.json').write_text(json.dumps(provenance,indent=2))
print('Encoder extraction verified',provenance['extracted_sha256'],flush=True)
zpath=ROOT/'data/adjust_bottle_clean50.zip'
if not zpath.exists():
 zpath.write_bytes(get(f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{DATAREV}/'+provenance['data_path']))
assert hashlib.sha256(zpath.read_bytes()).hexdigest()==provenance['data_expected_sha256']
target=ROOT/'data/adjust_bottle';target.mkdir(exist_ok=True)
with zipfile.ZipFile(zpath) as z:
 for member in z.infolist():
  p=(target/member.filename).resolve();assert p.is_relative_to(target.resolve())
 z.extractall(target)
print('Dataset hash verified and extracted',len(list(target.rglob('*.hdf5'))),'HDF5 files',flush=True)
