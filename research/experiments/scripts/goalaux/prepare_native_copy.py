"""Bounded copy into a NEW Group2 directory; never edit the original env."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path
R=Path('/data02/zifanz4/openwam-experiments');S=Path('/home/zifanz4/openwam-experiments/studies/goalaux-20260922');HOST='zifanz4@cm002.csl.illinois.edu';DEST='/vol13/zifanz4/openwam-experiments/goalaux-native'
def write(n,x):(S/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (S/'native-copy-started.json').exists()
 now=datetime.datetime.now().astimezone();deadline=datetime.datetime.fromisoformat('2026-09-23T01:15:12-05:00');assert now<deadline
 for p in [S/'paused.json',S.parent/'layers-20260922/paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),p
 sources=[R/'benchmark-env',R/'benchmarks/RoboTwin'];sizes=[int(subprocess.check_output(['du','-s','--apparent-size','--block-size=1',str(p)],text=True).split()[0]) for p in sources];assert sum(sizes)<20*1024**3,sizes
 native=R/'benchmarks/RoboTwin';paths=list((native/'envs').rglob('*.py'))+list((native/'task_config').rglob('*.yml'))+[R/'OpenWAM/benchmarks/robotwin/eval_policy_wrapper.py']
 hashes={str(p.relative_to(R)):sha(p) for p in paths if p.is_file()}
 write('native-source-manifest.json',{'recorded_at':now.isoformat(),'apparent_bytes':sum(sizes),'source_sha256':hashes,'reference':'https://github.com/haosulab/SAPIEN/discussions/99','binding_gate':'Runtime PCI/device check required despite documentary CUDA_VISIBLE_DEVICES guidance','scope':'Same native sources and configs; isolated environment relocation only'})
 script="from pathlib import Path\nimport shutil\np=Path('/vol13/zifanz4/openwam-experiments/goalaux-native')\nassert not p.exists()\nassert shutil.disk_usage(p.parent).free>25*1024**3\np.mkdir()\n(p/'copy-claim').mkdir()\n"
 subprocess.run(['ssh','-o','BatchMode=yes',HOST,'python3','-'],input=script,text=True,check=True,timeout=30)
 write('native-copy-started.json',{'started_at':now.isoformat(),'scope':'two new directories only','source':list(map(str,sources)),'destination':DEST,'apparent_bytes':sum(sizes),'wall_seconds_cap':900,'deadline':deadline.isoformat()})
 start=time.monotonic()
 for src,folder in zip(sources,['benchmark-env','RoboTwin']):
  seconds=min(900-int(time.monotonic()-start),int((deadline-datetime.datetime.now().astimezone()).total_seconds()));assert seconds>0
  with (S/f'native-copy-{folder}.log').open('w') as log:
   p=subprocess.run(['timeout','--signal=TERM','--kill-after=20s',str(seconds),'rsync','-a','--timeout=60','--stats',str(src)+'/',f'{HOST}:{DEST}/{folder}/'],stdout=log,stderr=subprocess.STDOUT)
  write('native-copy-progress.json',{'folder':folder,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-start})
  if p.returncode:raise RuntimeError(f'Copy failed {folder}, retained partial new copy, exit{p.returncode}')
 write('native-copy-complete.json',{'completed_at':datetime.datetime.now().astimezone().isoformat(),'elapsed_seconds':time.monotonic()-start,'original_environment_unchanged':True,'new_copy_usable':False,'next':'Verify sources; relocate only copied curobo.pth and script shebang paths; then bounded imports/render/PCI preflight before scenes'})
 print('NATIVE_COPY_COMPLETE_PENDING_RELOCATION',flush=True)
if __name__=='__main__':
 try:main();code=0
 except Exception as exc:write('native-copy-failure.json',{'time':datetime.datetime.now().astimezone().isoformat(),'error':repr(exc)});raise
 finally:
  import sys
  (S/'native-copy-exit-code.txt').write_text('1\n' if sys.exc_info()[0] else '0\n')
