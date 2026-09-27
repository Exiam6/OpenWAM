"""Finite per-GPU evaluation list, same native server/settings and every frozen condition."""
import json,sys
from pipeline_common import *
index=int(sys.argv[1]);plan=json.loads((S/'pipeline-plan.json').read_text());frozen(plan['code_sha256']);guard();assert socket.gethostname().split('.')[0]=='cm009' and index in range(1,7)
out=E/'evaluation-slots'/str(index);out.mkdir(parents=True,exist_ok=True);assert not(out/'started.json').exists()
code=1;server=None;log=None
write(out/'started.json',{'time':now().isoformat(),'index':index,'groups':plan['evaluation_groups'][str(index)]})
try:
 gate=json.loads((S/'evaluation-ready.json').read_text());assert gate['passed'];frozen(gate['frozen_files']);lock,row=reserve(index);write(out/'resource.json',{'time':now().isoformat(),'gpu':row})
 for group in plan['evaluation_groups'][str(index)]:
  route,seed=group['route'],group['seed'];name=f'{route}-seed{seed}';gout=E/'evaluation'/name;gout.mkdir(parents=True,exist_ok=True);assert not (gout/'started.json').exists()
  ckpts=list((E/'training'/name/'output').glob('*/checkpoint_step_12000.safetensors'));assert len(ckpts)==1
  assert sha(ckpts[0])==gate['checkpoint_sha256'][name]
  write(gout/'started.json',{'time':now().isoformat(),'checkpoint':str(ckpts[0]),'checkpoint_sha256':gate['checkpoint_sha256'][name]})
  policy,sim=environments(row,gout);server,log=start_server(ckpts[0].parent,18850+index,gout,policy)
  try:
   for task in ['adjust_bottle','handover_block','place_object_basket']:
    for condition in ['clean','gaussian_sigma_0.04','gaussian_sigma_0.10','head_camera_yaw_5deg']:
     dest=gout/task/condition;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'started.json').exists()
     command=[str(R/'benchmark-env/bin/python'),'-u',str(P/'evaluate_policy.py'),'--task',task,'--route',route,'--seed',str(seed),'--condition',condition,'--port',str(18850+index),'--output',str(dest)]
     result=run_child(command,sim,dest/'run.log',9000);write(dest/'exit.json',{'time':now().isoformat(),'returncode':result});assert result==0,'Failed evaluation retained; no automatic replay'
   dest=E/'action-diagnostics'/name;dest.mkdir(parents=True,exist_ok=True)
   command=[str(R/'benchmark-env/bin/python'),'-u',str(P/'action_diagnostics.py'),'--route',route,'--seed',str(seed),'--port',str(18850+index),'--output',str(dest)]
   result=run_child(command,sim,dest/'run.log',3600);write(dest/'exit.json',{'time':now().isoformat(),'returncode':result});assert result==0,'Failed diagnostic retained; no automatic replay'
  finally:terminate_owned(server);log.close();server=None;log=None
  write(gout/'exit.json',{'time':now().isoformat(),'returncode':0})
 code=0
finally:
 if server:terminate_owned(server)
 if log:log.close()
 write(out/'exit.json',{'time':now().isoformat(),'returncode':code})
sys.exit(code)
