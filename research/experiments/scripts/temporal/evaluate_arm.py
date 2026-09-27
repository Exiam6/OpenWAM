"""One finite arm/seed evaluation, gated by a development-only simulator rollout."""
import datetime,hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from run_owned import guard,write
R=Path('/data02/zifanz4/openwam-experiments');N=R/'temporal-20260926';P=Path('/home/zifanz4/openwam-experiments');S=P/'studies/temporal-20260926';SCRIPT=P/'scripts/temporal'
def stop(proc):
 if proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=20)
def child(cmd,env,log,seconds):
 guard()
 with log.open('x') as f:
  proc=subprocess.Popen(cmd,env=env,cwd=N/'OpenWAM',stdout=f,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
  try:
   while proc.poll() is None:
    guard();assert time.monotonic()-start<seconds,cmd;time.sleep(5)
   assert proc.returncode==0,(proc.returncode,str(log))
  finally:stop(proc)
def main(arm,seed):
 out=Path(os.environ['OPENWAM_RUN_DIR']);protocol=json.loads((S/'protocol.json').read_text());assert arm in ['mean','learned'] and seed in [42,43,44]
 audit=json.loads((S/'training-audit.json').read_text());assert audit['passed']
 weights=list((N/'training'/f'{arm}-seed{seed}'/'output').glob('*/checkpoint_step_4000.safetensors'));assert len(weights)==1;checkpoint=weights[0]
 cohort=json.loads((S/'fresh-cohort-integrity.json').read_text());assert cohort['passed']
 for file,sha in cohort['files_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha,file
 frozen=json.loads((S/'evaluation-code-manifest.json').read_text())
 for file,sha in frozen.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha,file
 uuid=os.environ['CUDA_VISIBLE_DEVICES'];pci=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip()
 port=19260+(0 if arm=='mean' else 1);probe=socket.socket();probe.bind(('127.0.0.1',port));probe.close()
 env=dict(os.environ);sim=dict(env,EXPECTED_RENDER_PCI=pci,WAM_ROOT=str(R),ROBOTWIN_PATH=str(R/'benchmarks/RoboTwin'),ROBOTWIN_RUNTIME_ROOT=str(out/'runtime'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1');sim.pop('PYTHONPATH',None)
 with (out/'server.log').open('x') as f:
  server=subprocess.Popen([str(R/'policy-env/bin/python'),'-u',str(SCRIPT/'serve_policy.py'),'--checkpoint',str(checkpoint),'--port',str(port),'--log',str(out/'episode-rng.jsonl')],env=env,cwd=N/'OpenWAM',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(out/'server-pid.json',{'pid':server.pid,'time':datetime.datetime.now().astimezone().isoformat()});begin=time.monotonic()
  try:
   while 'ENDTOEND_SERVER_READY' not in (out/'server.log').read_text(errors='replace'):
    guard();assert server.poll() is None;assert time.monotonic()-begin<600;time.sleep(5)
   smoke=out/'development';smoke.mkdir(exist_ok=False)
   child([str(R/'benchmark-env/bin/python'),'-u',str(SCRIPT/'simulation_preflight.py'),'--port',str(port),'--output',str(smoke)],sim,smoke/'run.log',1800)
   smoke_result=json.loads((smoke/'result.json').read_text());assert smoke_result['passed'] and smoke_result['tasks'][0]['full_development_rollout']['completed']
   write(S/f'simulation-preflight-{arm}-{seed}.json',{'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),'checkpoint':str(checkpoint),'development_result':str(smoke/'result.json'),'code_sha256':frozen})
   for task in protocol['evaluation']['tasks']:
    for condition in protocol['evaluation']['conditions']:
     dest=out/task/condition;dest.mkdir(parents=True,exist_ok=False)
     child([str(R/'benchmark-env/bin/python'),'-u',str(SCRIPT/'evaluate_policy.py'),'--task',task,'--route',arm,'--seed',str(seed),'--condition',condition,'--port',str(port),'--output',str(dest)],sim,dest/'run.log',12600)
     result=json.loads((dest/'summary.json').read_text());assert result['complete'] and result['episodes']==20
   write(out/'complete.json',{'complete':True,'episodes':180,'arm':arm,'seed':seed,'time':datetime.datetime.now().astimezone().isoformat()})
  finally:stop(server)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
