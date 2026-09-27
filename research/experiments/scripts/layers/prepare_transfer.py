import hashlib,json,torch,h5py,numpy as np
from pathlib import Path
R=Path('/vol13/zifanz4/openwam-experiments/results/layers-20260922');root=R.parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
p=R/'transfer';p.mkdir(exist_ok=True)
assert not (p/'manifest.json').exists(),'Transfer package already prepared; do not overwrite'
rows=json.loads((R/'readout/manifest.json').read_text());first=rows[:8];assert all(r['split']=='train' for r in first);assert len({(r['task'],r['episode']) for r in first})==1
shard=torch.load(R/'features'/first[0]['task']/f"episode{first[0]['episode']}.pt",map_location='cpu',weights_only=False);assert shard['manifest'][:8]==first
with h5py.File(root/first[0]['path'],'r') as f:images=[torch.from_numpy(np.frombuffer(bytes(f['observation/head_camera/rgb'][r['frame']]),np.uint8).copy()) for r in first]
v=np.load(R/'readout/vectors.npz');names=[s+'_'+k for s in ['L12','L6','K4'] for k in ['raw','pca','svae42','svae43','svae44']]
torch.save({'rows':first,'encoded_images':images,'features':{s:shard['features'][s][:8,:,:1].clone() for s in ['L12','L6','K4']},'vectors':{k:torch.from_numpy(v[k][:8].copy()) for k in names}},p/'parity-fixture.pt')
identity=[]
for task in ['adjust_bottle','handover_block','place_object_basket']:
 for file in sorted((root/'data'/task).rglob('*.hdf5')):
  with h5py.File(file,'r') as f:
   raw=bytes(f['observation/head_camera/rgb'][0]);n=len(f['observation/head_camera/rgb'])
  identity.append({'task':task,'path':str(file.relative_to(root)),'hdf5_sha256':sha(file),'first_head_encoded_sha256':hashlib.sha256(raw).hexdigest(),'frames':n})
assert len(identity)==150
(p/'old-corpus-identities.json').write_text(json.dumps(identity,indent=2)+'\n')
files=['transfer/parity-fixture.pt','transfer/old-corpus-identities.json','readout/coefficients.npz','readout/mlp/models.pt','readout/manifest.json']
for source in ['L12','L6','K4']:files.extend([f'{source}/pca.pt']+[f'{source}/seed{s}/final.pt' for s in [42,43,44]])
manifest={'sha256':{x:sha(R/x) for x in files},'encoder_sha256':sha(root/'assets/dinov3-study/encoder.safetensors'),'parity_fixture':'first8trainingframes; no new inference or model fitting','existing_corpus_episodes':150}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(p/'files.txt').write_text('\n'.join(files+['transfer/manifest.json'])+'\n');print(json.dumps({'files':len(files),'bytes':sum((R/f).stat().st_size for f in files)}),flush=True)
