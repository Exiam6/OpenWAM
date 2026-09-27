"""One finite dependency chain, no timer: train+collect+smoke -> all paired eval -> audit."""
import json,sys,time
from pipeline_common import *
plan=json.loads((S/'pipeline-plan.json').read_text());frozen(plan['code_sha256']);guard();out=E/'pipeline';out.mkdir(exist_ok=True);assert not(out/'started.json').exists();write(out/'started.json',{'time':now().isoformat(),'kind':'finite_dependency_chain','plan_sha256':sha(S/'pipeline-plan.json')})
code=1
try:
 deadline=datetime.datetime.fromisoformat(plan['dependency_wait_deadline']);groups=[x for v in plan['evaluation_groups'].values() for x in v]
 while True:
  guard();assert now()<deadline,'Prespecified dependency deadline; no extension';ready=True
  for g in groups:
   p=E/'training'/f'{g["route"]}-seed{g["seed"]}'/'exit.json'
   if p.exists():assert json.loads(p.read_text())['returncode']==0,('Training failed',str(p))
   else:ready=False
  for slot in [5,7]:
   p=E/'collection-slots'/str(slot)/'exit.json'
   if p.exists():assert json.loads(p.read_text())['returncode']==0,('Collection failed',str(p))
   else:ready=False
  p=E/'simulation-preflight/exit.json'
  if p.exists():assert json.loads(p.read_text())['returncode']==0,'Simulator preflight failed'
  else:ready=False
  write(out/'progress.json',{'time':now().isoformat(),'stage':'waiting_for_all_training_collection_smoke','all_ready':ready})
  if ready:break
  time.sleep(30)
 frozen(plan['code_sha256']);assert json.loads((S/'simulation-preflight.json').read_text())['passed']
 result=run_child([str(R/'benchmark-env/bin/python'),str(P/'verify_fresh_cohort.py')],os.environ.copy(),out/'cohort-integrity.log',1800);assert result==0
 import hashlib
 checkpoint_hashes={};freeze=dict(plan['code_sha256']);streams={}
 for seed in [42,43,44]:
  digests={route:sha(E/'training'/f'{route}-seed{seed}'/'batch-stream.jsonl') for route in ['svae','pca','wan']};assert len(set(digests.values()))==1,('unpaired training',seed,digests);streams[seed]=digests
 for g in groups:
  name=f'{g["route"]}-seed{g["seed"]}';root=E/'training'/name;metrics=[json.loads(x) for x in (root/'metrics-rank0.jsonl').read_text().splitlines()];last=metrics[-1];assert last['global_step']==12000 and last['opt_step']==6000
  assert len(metrics)==12000 and len((root/'batch-stream.jsonl').read_text().splitlines())==12000
  ckpt=list((root/'output').glob('*/checkpoint_step_12000.safetensors'));assert len(ckpt)==1;checkpoint_hashes[name]=sha(ckpt[0])
  for name_ in ['config.yaml','normalization_stats.npy']:
   p=ckpt[0].parent/name_;freeze[str(p)]=sha(p)
 for task in ['adjust_bottle','handover_block','place_object_basket']:
  p=E/'scenes'/task/'manifest.json';freeze[str(p)]=sha(p)
 write(S/'evaluation-ready.json',{'passed':True,'time':now().isoformat(),'checkpoint_sha256':checkpoint_hashes,'training_streams':streams,'frozen_files':freeze})
 launched=[]
 for slot in range(1,7):
  command=f"tmux new-session -d -s ow-e2e-eval-{slot} 'timeout --signal=TERM --kill-after=30s 232800 python3 {P}/evaluation_queue.py {slot} > {E}/evaluation-queue-{slot}.log 2>&1'"
  result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','zifanz4@cm009.csl.illinois.edu',command],timeout=30);assert result.returncode==0,('Queue launch failed',slot);launched.append(slot);write(out/'launched.json',{'time':now().isoformat(),'slots':launched})
 end=now()+datetime.timedelta(seconds=233000)
 while True:
  guard();assert now()<end,'Prespecified evaluation queue window elapsed';done=[]
  for slot in range(1,7):
   p=E/'evaluation-slots'/str(slot)/'exit.json'
   if p.exists():assert json.loads(p.read_text())['returncode']==0,('Evaluation failed',slot);done.append(slot)
  write(out/'progress.json',{'time':now().isoformat(),'stage':'paired_evaluation','completed_slots':done})
  if len(done)==6:break
  time.sleep(30)
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 for script in ['analyze_closed_loop.py','audit_closed_loop.py','summarize_diagnostics.py']:
  result=run_child([str(R/'benchmark-env/bin/python'),str(P/script)],env,out/f'{script}.log',3600);assert result==0,script
 write(out/'progress.json',{'time':now().isoformat(),'stage':'complete_audited_native295M_evaluation','next':'Assess the fixed primary and all secondary results; freeze scale-transfer experiment prospectively if supported. No automatic5Bbenefit claim.'});code=0
finally:
 write(out/'exit.json',{'time':now().isoformat(),'returncode':code})
 subprocess.run(['python3',str(P/'update_status.py')],timeout=60)
sys.exit(code)
