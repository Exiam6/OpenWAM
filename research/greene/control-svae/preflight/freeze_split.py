"""Control-SVAE preflight gate: freeze the fit/dev split of the 105 training episodes.

PLAN rule: per task, fixed episode hash, 28 fit / 7 dev, seed 20260926; adjacent frames,
same trajectory and same scene variant stay in one partition. The split is episode-level, so
frames and trajectories never straddle partitions. Scene variants carry no seed in the hdf5,
so they are checked by near-duplicate first observations (head-camera image + initial joint
vector) within each task; a near-duplicate pair across partitions fails the gate (no silent
reassignment). Read-only; no fitting, no labels used for the assignment.
"""
import glob,hashlib,io,json,datetime,re,sys
import h5py,numpy as np
from PIL import Image
SEED=20260926;N_DEV=7;N_FIT=28
T='/scratch/zz4330/openwam-runtime/endtoend-root/results/layers-20260922/native-preflight/train-data'
IMG_THR=2.0;JOINT_THR=1e-3  # fixed before looking: mean abs 32x32 gray diff (0-255), max abs joint diff (rad)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def key(task,ep):return hashlib.sha256(f'{SEED}:{task}:episode{ep}'.encode()).hexdigest()
fs=sorted(glob.glob(T+'/*/aloha-agilex_clean_50/data/episode*.hdf5'));assert len(fs)==105,len(fs)
tasks={}
for f in fs:
 t=f[len(T)+1:].split('/')[0];ep=int(re.search(r'episode(\d+)\.hdf5$',f).group(1))
 with h5py.File(f,'r') as h:
  img=np.asarray(Image.open(io.BytesIO(h['observation/head_camera/rgb'][0])).convert('L').resize((32,32),Image.BILINEAR),dtype=np.float32)
  q0=h['joint_action/vector'][0];n=len(h['joint_action/vector'])
 tasks.setdefault(t,[]).append({'episode':ep,'path':f,'sha256':sha(f),'frames':n,'valid_t4_pairs':n-4,'excluded_tail_frames':4,'key':key(t,ep),'_img':img,'_q0':q0})
out={'time':datetime.datetime.now().astimezone().isoformat(timespec='seconds'),'gate':'control-SVAE preflight: fit/dev split freeze',
 'scope':'read-only; 105 training episodes; assignment uses only task name + episode index','rule':{'seed':SEED,'per_task':{'fit':N_FIT,'dev':N_DEV},
 'assignment':"sort episodes of each task by sha256(f'{seed}:{task}:episode{idx}') ascending; first 7 = dev, remaining 28 = fit",
 'unit':'whole episode (all frames and t+4 windows of a trajectory share its partition)',
 'scene_variant_check':{'image':'head_camera frame 0, gray 32x32 bilinear, mean abs diff','image_threshold':IMG_THR,'joint':'frame-0 joint_action/vector max abs diff','joint_threshold':JOINT_THR,'near_duplicate':'image diff < threshold'}},
 'tasks':{},'near_duplicate_pairs':[],'cross_partition_near_duplicates':0}
for t,eps in sorted(tasks.items()):
 eps.sort(key=lambda e:e['key'])
 for i,e in enumerate(eps):e['partition']='dev' if i<N_DEV else 'fit'
 dmin=float('inf')
 for i in range(len(eps)):
  for j in range(i+1,len(eps)):
   a,b=eps[i],eps[j];d=float(np.abs(a['_img']-b['_img']).mean());dq=float(np.abs(a['_q0']-b['_q0']).max());dmin=min(dmin,d)
   if d<IMG_THR:
    out['near_duplicate_pairs'].append({'task':t,'episodes':sorted([a['episode'],b['episode']]),'image_diff':round(d,3),'joint0_diff':round(dq,6),'same_partition':a['partition']==b['partition']})
    out['cross_partition_near_duplicates']+=a['partition']!=b['partition']
 rows=[{k:v for k,v in e.items() if not k.startswith('_')} for e in sorted(eps,key=lambda e:e['episode'])]
 out['tasks'][t]={'dev_episodes':sorted(e['episode'] for e in eps if e['partition']=='dev'),'fit_episodes':sorted(e['episode'] for e in eps if e['partition']=='fit'),
  'fit_valid_t4_pairs':sum(e['valid_t4_pairs'] for e in eps if e['partition']=='fit'),'dev_valid_t4_pairs':sum(e['valid_t4_pairs'] for e in eps if e['partition']=='dev'),
  'min_pairwise_frame0_image_diff':round(dmin,3),'episodes':rows}
out['total_valid_t4_pairs']=sum(v['fit_valid_t4_pairs']+v['dev_valid_t4_pairs'] for v in out['tasks'].values())
out['passed']=out['cross_partition_near_duplicates']==0 and all(len(v['dev_episodes'])==N_DEV and len(v['fit_episodes'])==N_FIT for v in out['tasks'].values())
out['caveat']='The initial S-VAE/policy may have seen all 105 episodes, so dev is for auxiliary-training development only, not an unseen generalization test (PLAN).'
open(sys.argv[1],'w').write(json.dumps(out,indent=1)+'\n')
print(json.dumps({k:(v if k!='tasks' else {t:{x:y for x,y in d.items() if x!='episodes'} for t,d in v.items()}) for k,v in out.items() if k not in('rule',)},indent=1))
