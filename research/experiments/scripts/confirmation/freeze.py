"""Freeze a separately authorized budget and all inputs before collection."""
import datetime,hashlib,json,py_compile,shutil,sys
from pathlib import Path
P=Path('/home/zifanz4/openwam-experiments');R=Path('/data02/zifanz4/openwam-experiments');O=R/'results/confirmation-20260923';S=P/'studies/confirmation-20260923'
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not (O/'freeze.json').exists()
window=json.loads((O/'window.json').read_text());plan=dict(window)
plan.update(status='frozen',frozen_at=datetime.datetime.now().astimezone().isoformat(),primary_candidate='L12_pca',primary_reference='Wan48_native_scale',primary_condition='noise0.10',co_primary=['first-close XY normalized MSE','short-state translation/gripper equal-group normalized MSE'],decision='both task-stratified paired two-sided95 bootstrap absolute-difference intervals strictly below0',candidate_selection='PCA48 selected using exposed prior60; those outcomes are not confirmation',bootstrap={'seed':20260922,'draws':2000,'unit':'scene','stratification':'20 scenes within each of3fixed tasks','SVAE':'average seed42/43/44 errors within scene before bootstrap'},conditions={'clean':1,'noise0.10':3,'noise0.04':3},readouts='frozen equal-noise goal/state coefficients; frozen original proprio-only control; zero refits',routes=['Wan48_native_scale','Wan48_per_token_layernorm','L12_raw','L12_pca','L12_svae42','L12_svae43','L12_svae44'],collection='first20 native expert-plan-and-replay successes among first40consecutive seeds per task; raw failures retained; narrow previously established null-grasp annotation only; unclassified infrastructure failure stops task with no retry',completeness='score only balanced60 with all3collector exits0, identity and hardware parity passed; incomplete cohort never scored; no replacement for short episode or missing firstclose label',frames='12 unique linspace(0,n-33,12,dtype=int); target st+4 minus st XYZ and absolute st+4 grippers; n>=44 required',pixels='cv2 IMREAD_COLOR preserving historical BGR byte interpretation; PIL bilinear320x384; Gaussian clipped0..255 then uint8 before native preprocessing',noise_seed='20260922 + scene_seed*100000 + frame*10 + draw; same draws for sigma.10/.04 as prior frozen pipeline',goal='initial head image, earliest gripper crossing >.5 to <=.5, left arm tie-break, XY endpose at crossing; not physical contact',mandatory_endpoints=['clean','translation','gripper','all3tasks','all7routes','proprio-only state'],gripper_rule='no noninferiority margin; improvement only if whole paired95 interval below0; report uncertainty and all task regressions',hardware_parity={'training_frames':'first initial training frame per task, indices0/35/70','normalized_RMS_max':.05,'normalized_coordinate_max':.5,'normalization':'per-task training vector std clipped1e-6','fixed_before_new_data':True},max_scene_seconds=600,collection_cleanup_seconds=30,limitations=['offline readouts only','no pixel decoder trained: not completeRAE','no policy/viewpoint/physical-contact gain','raw3072versuscompact192pooledinputs','pretraining not causally isolated','fixed known noise levels and tasks'])
for name in ['protocol.json']:
 write(O/name,plan);shutil.copy2(O/name,S/name)
for f in (P/'scripts/confirmation').glob('*.py'):
 py_compile.compile(str(f),doraise=True);shutil.copy2(f,R/'scripts/confirmation'/f.name)
assert json.loads((O/'cpu-check.json').read_text())['passed']
assert json.loads((O/'native-source-check.json').read_text())['passed']
assert json.loads((O/'seed-exclusion.json').read_text())['passed']
for t in plan['task_order']:
 assert json.loads((O/'preflight'/t/'result.json').read_text())['passed']
 assert json.loads((O/'preflight'/t/'exit.json').read_text())['exit_code']==0
files=set((R/'scripts/confirmation').glob('*.py'))|set((R/'scripts/goalaux').glob('*.py'))
for folder in ['rae-noise-control-20260922','rae-state-control-20260923']:
 prev=json.loads((R/'results'/folder/'freeze.json').read_text())
 for path,h in prev['files'].items():
  p=Path(path);assert sha(p)==h,path;files.add(p)
 files.update([R/'results'/folder/'coefficients.npz',R/'results'/folder/'selection.json',R/'results'/folder/'fit-complete.json'])
files.update((R/'scripts/layers').glob('*.py'))
files.update([R/'scripts/rae_noise_control.py',R/'scripts/rae_interface_preflight.py',R/'assets/wan22-vae-baseline/Wan2.2_VAE.pth',R/'assets/dinov3-study/encoder.safetensors',R/'results/layers-20260922/readout/coefficients.npz',R/'results/layers-20260922/initial-goal/constants.json'])
files.update([R/'results/layers-20260922/L12/pca.pt']+[R/f'results/layers-20260922/L12/seed{s}/final.pt' for s in [42,43,44]])
files.update([O/n for n in ['protocol.json','window.json','cpu-check.json','seed-exclusion.json','identity-reference.json','native-source-check.json']])
native=json.loads((P/'studies/goalaux-20260922/native-source-manifest.json').read_text())
files.update(R/k for k in native['source_sha256'])
# Pin encoder/decoder source and exact model configs as well as weights.
files.update((R/'OpenWAM/openwam/model/video_backbone/encoder').rglob('*.py'))
files.update((R/'OpenWAM/openwam/model/video_backbone/wan/models').glob('*.py'))
files.update((R/'assets/dinov3-study/dinov3').glob('*.json'))
freeze={'time':datetime.datetime.now().astimezone().isoformat(),'protocol_sha256':sha(O/'protocol.json'),'files':{str(p):sha(p) for p in sorted(files)},'new_confirmation_scenes_seen':0,'readout_refits':0}
write(O/'freeze.json',freeze);shutil.copy2(O/'freeze.json',S/'freeze.json')
for n in ['cpu-check.json','seed-exclusion.json','native-source-check.json']:
 shutil.copy2(O/n,S/n)
print('FROZEN',len(files),'files',freeze['time'])
