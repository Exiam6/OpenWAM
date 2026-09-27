"""Prospective cohort identity/labels; source and zero-noise parity gates."""
import datetime,hashlib,json,os,sys
from pathlib import Path
import cv2,h5py,numpy as np,torch
R=Path(os.environ['WAM_ROOT']);B=R/'results/goalaux-20260922';L=R/'results/layers-20260922';O=B/'confirmation';F=B/'fresh-v2'
os.environ['LAYER_RUN']=str(L);sys.path.insert(0,str(R/'scripts/layers'));import common as c
from readout_v2 import pool

def identity():
 assert not (O/'identity.json').exists();assert (F/'exit-code.txt').read_text().strip()=='0'
 m=json.loads((F/'manifest.json').read_text());assert m['complete'];old=json.loads((B/'previous-corpus-identities.json').read_text());assert len(old)==210
 full_old={q['hdf5_sha256'] for q in old};head_old={q['first_head_encoded_sha256'] for q in old};full=set();head=set();verified=[];labels=[];missing=[]
 for ti,task in enumerate(c.TASKS):
  attempts=[q for q in m['attempts'] if q['task']==task];start=800000+1000*ti;assert [q['seed'] for q in attempts]==list(range(start,start+len(attempts))) and len(attempts)<=40
  assert [q['seed'] for q in attempts if q.get('accepted')]==[q['seed'] for q in m['accepted'][task]] and len(m['accepted'][task])==20
  for rec in m['accepted'][task]:
   path=Path(rec['hdf5']);h=c.sha(path);assert h==rec['hdf5_sha256'] and h not in full|full_old;full.add(h)
   provenance=json.loads((path.parent.parent/'runtime-provenance.json').read_text());assert len(provenance)==2 and all(all(v=='CuroboPlanner' for v in q['planners'].values()) for q in provenance)
   assert all(q['renderer_pci']=='0000:a1:00.0' for q in provenance)
   with h5py.File(path,'r') as f:
    n=len(f['endpose/left_endpose']);assert n>=44 and len(np.unique(np.linspace(0,n-33,12,dtype=int)))==12
    hh=hashlib.sha256(bytes(f['observation/head_camera/rgb'][0])).hexdigest();assert hh not in head|head_old;head.add(hh)
    for key in ['observation/head_camera/rgb','observation/left_camera/rgb','observation/right_camera/rgb','endpose/right_endpose','endpose/left_gripper','endpose/right_gripper']:assert len(f[key])==n
    for key in ['endpose/left_endpose','endpose/right_endpose','endpose/left_gripper','endpose/right_gripper']:assert np.isfinite(f[key][:]).all()
    candidates=[]
    for ai,arm in enumerate(['left','right']):
     grip=f[f'endpose/{arm}_gripper'][:];idx=np.flatnonzero((grip[:-1]>.5)&(grip[1:]<=.5))+1
     if len(idx):candidates.append((int(idx[0]),ai,arm))
    if not candidates:missing.append({'task':task,'seed':rec['seed']})
    else:
     event,ai,arm=min(candidates);labels.append({'task':task,'seed':rec['seed'],'event_frame':event,'arm':arm,'target_xy':np.asarray(f[f'endpose/{arm}_endpose'][event,:2],dtype=np.float32).tolist(),'hdf5_sha256':h,'simultaneous_crossing':len(candidates)>1 and candidates[0][0]==candidates[1][0]})
   verified.append({'task':task,'seed':rec['seed'],'hdf5_sha256':h,'first_head_encoded_sha256':hh,'frames':n})
 c.write(O/'labels.json',{'records':labels,'missing':missing,'replacement_allowed':False});assert not missing,'Incomplete goal labels; no replacement or scoring'
 c.write(O/'identity.json',{'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),'records':verified,'previous_corpus_episodes':210,'episodes':60,'attempts':len(m['attempts']),'failed_setup_repair_count':1,'exact_duplicates':0,'complete_goal_labels':60,'limitations':['exact-byte identity only, not perceptual deduplication','unknown encoder pretraining overlap']})
 print('IDENTITY_LABELS_PASS',flush=True)

@torch.inference_mode()
def parity():
 assert not (O/'parity.json').exists();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 fixture=torch.load(L/'transfer/parity-fixture.pt',map_location='cpu',weights_only=False);enc=c.encoder();features=[];pixel_max=0
 for raw,row in zip(fixture['encoded_images'],fixture['rows']):
  assert hashlib.sha256(raw.numpy().tobytes()).hexdigest()==row['encoded_head_hashes'][0]
  im=c.resize_head(cv2.imdecode(raw.numpy(),cv2.IMREAD_COLOR));arr=np.array(im);zero=np.clip(arr.astype(np.float32)+np.zeros(arr.shape),0,255).astype(np.uint8);assert np.array_equal(arr,zero)
  from PIL import Image
  z=c.encode(enc,enc.preprocess_video([Image.fromarray(zero)]*9));features.append(z['L12'][0,:,:1].cpu().half())
 del enc;torch.cuda.empty_cache();features=torch.stack(features);delta=float((features-fixture['features']['L12']).abs().max());assert delta==0
 rows=json.loads((B/'readouts/training-manifest.json').read_text());lookup={(r['task'],r['episode'],r['frame']):i for i,r in enumerate(rows)};indices=[lookup[r['task'],r['episode'],r['frame']] for r in fixture['rows']];saved=np.load(B/'readouts/training-vectors.npz');errors={}
 for arm in ['reconstruction_only','goal_auxiliary']:
  for seed in [42,43,44]:
   ck=torch.load(B/'models'/arm/f'seed{seed}/final.pt',map_location='cpu',weights_only=False);model=c.SVAE(**ck['model_config']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['state_dict'])
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model.encode_mean(features.cuda().float())
   name=f'{arm}_svae{seed}';vec=pool(z);err=float(np.max(np.abs(vec-saved[name][indices])));errors[name]=err;assert err<=1e-6,(name,err);del model;torch.cuda.empty_cache()
 c.write(O/'parity.json',{'passed':True,'training_frames':len(indices),'encoder_feature_max_abs':delta,'codec_vector_max_abs':errors,'zero_noise_pixels_exact':True,'fresh_scores_observed':False})
 print('ZERO_NOISE_NEW_CODECS_PASS',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='identity':identity()
 elif sys.argv[1]=='parity':parity()
 else:raise ValueError(sys.argv[1])
