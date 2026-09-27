"""Finite dependency job: wait for collection GPU7 release, then training-side smoke once."""
import json,sys,time
from pipeline_common import *
out=E/'simulation-preflight';out.mkdir(exist_ok=True);assert not (out/'started.json').exists();config=json.loads((S/'pipeline-plan.json').read_text());frozen(config['code_sha256']);guard()
assert socket.gethostname().split('.')[0]=='cm001'
write(out/'started.json',{'time':now().isoformat(),'status':'waiting_for_collection_slot7','wait_deadline':config['smoke_wait_deadline']})
code=1;server=None;f=None
try:
 deadline=datetime.datetime.fromisoformat(config['smoke_wait_deadline'])
 while not (E/'collection-slots/7/exit.json').exists():
  guard();assert now()<deadline,'Collection dependency wait expired';time.sleep(20)
 time.sleep(5);lock,row=reserve(7);write(out/'resource.json',{'time':now().isoformat(),'gpu':row});policy,sim=environments(row,out)
 ckpts=list((E/'profiles-v3/pca/output').glob('*/checkpoint_step_32.safetensors'));assert len(ckpts)==1
 server,f=start_server(ckpts[0].parent,18847,out,policy)
 child_code=run_child([str(R/'benchmark-env/bin/python'),'-u',str(P/'simulation_preflight.py'),'--port','18847','--output',str(out)],sim,out/'client.log',1800)
 assert child_code==0 and json.loads((out/'result.json').read_text())['passed']
 write(S/'simulation-preflight.json',{'passed':True,'time':now().isoformat(),'artifact':str(out/'result.json'),'code_sha256':{str(P/n):sha(P/n) for n in ['evaluate_policy.py','serve_policy.py','simulation_preflight.py']},'heldout_policy_outcomes_used':False})
 code=0
finally:
 if server:terminate_owned(server)
 if f:f.close()
 write(out/'exit.json',{'time':now().isoformat(),'returncode':code})
sys.exit(code)
