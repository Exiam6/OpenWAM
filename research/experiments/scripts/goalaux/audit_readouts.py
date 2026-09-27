"""Independent saved-artifact audit; no fit, no held-out scoring."""
import hashlib,json,os,datetime
from pathlib import Path
import numpy as np,torch
R=Path(os.environ['WAM_ROOT']);B=R/'results/goalaux-20260922';O=B/'readouts'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'audit.json').exists()
complete=json.loads((O/'complete.json').read_text());assert (O/'exit-code.txt').read_text().strip()=='0'
for f,h in complete['sha256'].items():assert sha(O/f)==h,f
anchors=json.loads((O/'anchors.json').read_text())
for f,h in anchors['sha256'].items():assert sha(R/f)==h,f
rows=json.loads((O/'training-manifest.json').read_text());inputs=json.loads((B/'training-inputs.json').read_text());assert rows==inputs['rows'];assert len(rows)==1260 and all(r['split']=='train' for r in rows);assert len({(r['task'],r['episode']) for r in rows})==105
v=np.load(O/'training-vectors.npz');coef=np.load(O/'coefficients.npz');sel=json.loads((O/'selection.json').read_text());ck=torch.load(O/'mlp.pt',map_location='cpu',weights_only=False);meta=json.loads((O/'training.json').read_text());assert len(meta)==36 and len(sel)==54
initial=np.array([r['frame']==0 for r in rows]);assert np.array_equal(initial,v['initial']) and initial.sum()==105
names=[f'{a}_svae{s}' for a in ['reconstruction_only','goal_auxiliary'] for s in [42,43,44]]
taskvec=np.array([r['task'] for r in rows]);epvec=np.array([r['episode'] for r in rows]);checks=[]
for endpoint in ['state','goal']:
 for task in sorted(set(taskvec)):
  mask=(taskvec==task)&(initial if endpoint=='goal' else True);eps=epvec[mask];expected_folds=[q.tolist() for q in np.array_split(np.random.default_rng(20260922).permutation(np.unique(eps)),5)]
  for name in names:
   key=f'{endpoint}/{task}/{name}';x=(v[name] if endpoint=='goal' else np.concatenate([v[name],v['state']],1))[mask];y=v['goal' if endpoint=='goal' else 'target'][mask];assert len(x)==(35 if endpoint=='goal' else 420)
   for group,sl in ([(0,slice(0,2))] if endpoint=='goal' else [(0,slice(0,6)),(1,slice(6,8))]):
    k=f'{key}/{group}';s=sel[k];assert s['folds']==expected_folds and s['selected']==min(s['all_scores'],key=lambda a:a['score'])['alpha'];assert [a['alpha'] for a in s['all_scores']]==[1e-4,.01,1.,100.]
    for a in s['all_scores']:assert np.isclose(a['score'],np.mean(a['fold_scores']))
    m={a:coef[k+'/'+a] for a in ['xm','xs','ym','ys','w']};xx=x.astype(float);yy=y[:,sl].astype(float)
    for a,b in [('xm',xx.mean(0)),('xs',xx.std(0).clip(1e-6)),('ym',yy.mean(0)),('ys',yy.std(0).clip(1e-6))]:assert np.allclose(m[a],b,rtol=1e-12,atol=1e-12),(k,a)
    z=(xx-m['xm'])/m['xs']/np.sqrt(xx.shape[1]);t=(yy-m['ym'])/m['ys'];rhs=z.T@t;res=(z.T@z+s['selected']*np.eye(z.shape[1]))@m['w']-rhs;relative=float(np.linalg.norm(res)/max(1,np.linalg.norm(rhs)));assert relative<1e-8,(k,relative)
    checks.append({'readout':k,'ridge_normal_equation_relative_residual':relative})
   state=ck['states'][key];scales=[a.numpy() for a in ck['scales'][key]];assert all(torch.isfinite(q).all() for q in state.values());assert meta[key]['train_episodes']==35 and meta[key]['steps']==500 and meta[key]['seed']==42
   for a,b in zip(scales,[x.mean(0),x.std(0).clip(1e-6),y.mean(0),y.std(0).clip(1e-6)]):assert np.array_equal(a,b)
   net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,y.shape[1]));net.load_state_dict(state);net.eval();xx=(x[:7]-scales[0])/scales[1]
   with torch.no_grad():pred=net(torch.from_numpy(xx)).numpy()
   manual=xx
   for idx in [0,2,4]:
    manual=manual@state[f'{idx}.weight'].numpy().T+state[f'{idx}.bias'].numpy()
    if idx<4:manual=np.maximum(manual,0)
   assert np.allclose(pred,manual,rtol=1e-4,atol=3e-5),key
out={'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),'training_episodes':105,'training_rows':1260,'ridge_groups_checked':54,'mlp_models_checked':36,'anchor_hashes_verified':len(anchors['sha256']),'max_ridge_normal_equation_relative_residual':max(a['ridge_normal_equation_relative_residual'] for a in checks),'saved_mlp_torch_numpy_parity':True,'no_refit_or_heldout_scoring':True,'complete_manifest_sha256':sha(O/'complete.json'),'auditor_sha256':sha(Path(__file__)),'checks':checks}
(O/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
