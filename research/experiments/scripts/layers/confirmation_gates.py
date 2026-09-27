"""Pre-score frozen-asset, host-parity and fresh-data identity gates."""
import argparse,hashlib,json,os
from pathlib import Path
import cv2,h5py,numpy as np,torch
import common as c
from readout_v2 import visual
R=Path(os.environ['LAYER_RUN'])
def parity():
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    frozen=json.loads((R/'confirmation-freeze.json').read_text())
    for p,h in frozen['sha256'].items():assert c.sha(R/p)==h,p
    assert c.sha(c.ROOT/'assets/dinov3-study/encoder.safetensors')==frozen['encoder_sha256']
    fixture=torch.load(R/'transfer/parity-fixture.pt',map_location='cpu',weights_only=False);enc=c.encoder();out={s:[] for s in c.SOURCES}
    for raw,row in zip(fixture['encoded_images'],fixture['rows']):
        assert hashlib.sha256(raw.numpy().tobytes()).hexdigest()==row['encoded_head_hashes'][0]
        im=c.resize_head(cv2.imdecode(raw.numpy(),cv2.IMREAD_COLOR));z=c.encode(enc,enc.preprocess_video([im]*9))
        for s in c.SOURCES:out[s].append(z[s][0,:,:1].cpu().half())
    del enc;torch.cuda.empty_cache();features={s:torch.stack(out[s]) for s in c.SOURCES};fd={s:float((features[s]-fixture['features'][s]).abs().max()) for s in c.SOURCES};vd={}
    for s in c.SOURCES:
        vectors=visual(features[s],s)
        for key,arr in vectors.items():vd[key]=float(np.max(np.abs(arr-fixture['vectors'][key].numpy())))
    record={'passed':max(fd.values())==0 and max(vd.values())<=1e-6,'encoder_feature_max_abs':fd,'reducer_vector_max_abs':vd,'features_required_exact':True,'vector_absolute_tolerance':1e-6,'training_frames':8,'no_fresh_scores':True}
    c.write(R/'confirmation-parity.json',record);assert record['passed'],record;print(json.dumps(record),flush=True)
def identity():
    assert (R/'fresh/continuation-exit-code.txt').read_text().strip()=='0'
    m=json.loads((R/'fresh/manifest-v2.json').read_text());assert m['complete'] and all(len(m['accepted'][t])==20 for t in c.TASKS)
    old=json.loads((R/'transfer/old-corpus-identities.json').read_text());oldfull={r['hdf5_sha256'] for r in old};oldhead={r['first_head_encoded_sha256'] for r in old}
    full=set();head=set();seeds=set();verified=[]
    for ti,task in enumerate(c.TASKS):
        alltask=[r for r in m['attempts'] if r['task']==task];ss=[r['seed'] for r in alltask];assert ss==list(range(600000+1000*ti,600000+1000*ti+len(ss))) and len(ss)<=40
        expected=[r['seed'] for r in alltask if r.get('accepted')][:20];assert expected==[r['seed'] for r in m['accepted'][task]]
        for r in m['accepted'][task]:
            assert r['seed'] not in seeds;seeds.add(r['seed']);p=Path(r['hdf5']);h=c.sha(p);assert h==r['hdf5_sha256'] and h not in full and h not in oldfull;full.add(h)
            with h5py.File(p,'r') as f:
                n=len(f['endpose/left_endpose']);assert n>=44 and len(np.unique(np.linspace(0,n-33,12,dtype=int)))==12
                raw=bytes(f['observation/head_camera/rgb'][0]);hh=hashlib.sha256(raw).hexdigest();assert hh not in head and hh not in oldhead;head.add(hh)
                for key in ['observation/head_camera/rgb','observation/left_camera/rgb','observation/right_camera/rgb','endpose/right_endpose','endpose/left_gripper','endpose/right_gripper']:assert len(f[key])==n
                for key in ['endpose/left_endpose','endpose/right_endpose','endpose/left_gripper','endpose/right_gripper']:assert np.isfinite(f[key][:]).all()
            verified.append({'task':task,'seed':r['seed'],'hdf5_sha256':h,'first_head_encoded_sha256':hh,'frames':n})
    c.write(R/'confirmation-identity.json',{'passed':True,'accepted':len(verified),'total_attempts':len(m['attempts']),'existing_corpus_episodes':len(old),'fixed_consecutive_attempt_order':True,'exact_hdf5_and_first_image_no_duplicates':True,'records':verified,'limits':['exact byte identity check; not perceptual scene deduplication','unknown external pretraining corpus not audited']});print('Identity gates passed60fresh against150existing episodes',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['parity','identity']);a=p.parse_args()
    if a.mode=='parity':parity()
    else:identity()
