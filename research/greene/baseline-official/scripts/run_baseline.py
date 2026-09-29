"""Finite official-checkpoint baseline: cohort check, server, development gate, 3 x 20 clean scenes."""
import datetime,hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
REPO=Path('/scratch/zz4330/OpenWAM');B=REPO/'research/greene/baseline-official';SCRIPT=B/'scripts'
RUNTIME=Path('/scratch/zz4330/openwam-runtime');ASSETS=RUNTIME/'assets-source'
POLICY_PY='/scratch/zz4330/conda/envs/openwam-policy/bin/python';BENCH_PY='/scratch/zz4330/conda/envs/openwam-bench/bin/python'
CKPT=REPO/'assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full/checkpoint_step_118655.safetensors'
COHORT=REPO/'research/experiments/studies/temporal-recovery-v2-20260927/fresh-cohort-integrity.json'
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<24),b''):h.update(b)
 return h.hexdigest()
def stop(proc):
 if proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=20)
def child(cmd,env,log,seconds):
 with log.open('x') as f:
  proc=subprocess.Popen(cmd,env=env,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
  try:
   while proc.poll() is None:
    assert time.monotonic()-start<seconds,cmd;time.sleep(5)
   assert proc.returncode==0,(proc.returncode,str(log))
  finally:stop(proc)
def main():
 out=Path(os.environ['OPENWAM_RUN_DIR']);protocol=json.loads((B/'protocol.json').read_text())
 # Sealed cohort: GPU02 paths remapped to the byte-exact Greene copy.
 cohort=json.loads(COHORT.read_text());assert cohort['passed'];files={}
 for file,want in cohort['files_sha256'].items():
  local=Path(file.replace('/data02/zifanz4/openwam-experiments/',str(ASSETS)+'/'));assert sha(local)==want,local;files[str(local)]=want
 write(out/'cohort-integrity.json',{'passed':True,'time':now(),'source':str(COHORT),'files_sha256':files})
 code={str(p):sha(p) for p in sorted(SCRIPT.glob('*.py'))+[B/'protocol.json']}
 write(out/'run-record.json',{'time':now(),'host':socket.gethostname(),'slurm_job':os.environ.get('SLURM_JOB_ID'),'checkpoint':str(CKPT),'checkpoint_bytes':CKPT.stat().st_size,'code_sha256':code,'protocol':protocol})
 pci=subprocess.check_output(['nvidia-smi','--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip().splitlines();assert len(pci)==1,pci
 port=19400+int(os.environ.get('SLURM_JOB_ID','0'))%500;probe=socket.socket();probe.bind(('127.0.0.1',port));probe.close()
 env=dict(os.environ,PYTHONPATH=str(REPO),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
 sim=dict(env,EXPECTED_RENDER_PCI=pci[0],WAM_ROOT=str(RUNTIME),ROBOTWIN_PATH=str(RUNTIME/'benchmarks/RoboTwin'),ROBOTWIN_RUNTIME_ROOT=str(out/'runtime'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1');sim.pop('PYTHONPATH',None)
 with (out/'server.log').open('x') as f:
  server=subprocess.Popen([POLICY_PY,'-u',str(SCRIPT/'serve_policy.py'),'--checkpoint',str(CKPT),'--port',str(port),'--log',str(out/'episode-rng.jsonl')],env=env,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  begin=time.monotonic()
  try:
   while 'ENDTOEND_SERVER_READY' not in (out/'server.log').read_text(errors='replace'):
    assert server.poll() is None,'server exited before ready';assert time.monotonic()-begin<1800,'server not ready in 1800 s';time.sleep(5)
   smoke=out/'development';smoke.mkdir(exist_ok=False)
   child([BENCH_PY,'-u',str(SCRIPT/'simulation_preflight.py'),'--port',str(port),'--output',str(smoke)],sim,smoke/'run.log',3600)
   result=json.loads((smoke/'result.json').read_text());assert result['passed'] and result['tasks'][0]['full_development_rollout']['completed']
   write(out/'simulation-preflight-official-42.json',{'passed':True,'time':now(),'checkpoint':str(CKPT),'development_result':str(smoke/'result.json'),'code_sha256':code})
   summaries={}
   for task in protocol['evaluation']['tasks']:
    manifest=json.loads((ASSETS/'temporal-20260926/scenes'/task/'manifest.json').read_text());seeds=[int(x['seed']) for x in manifest['accepted']];assert len(seeds)==20
    dest=out/task/'clean';dest.mkdir(parents=True,exist_ok=False)
    child([BENCH_PY,'-u',str(SCRIPT/'evaluate_policy.py'),'--scene-seeds',json.dumps(seeds),'--task',task,'--route','official','--seed','42','--condition','clean','--port',str(port),'--output',str(dest)],sim,dest/'run.log',20*660)
    s=json.loads((dest/'summary.json').read_text());summaries[task]={k:s[k] for k in ['episodes','successes','technical_failures','complete']};print(task,summaries[task],flush=True)
   write(out/'complete.json',{'time':now(),'tasks':summaries,'successes':sum(v['successes'] for v in summaries.values()),'episodes':sum(v['episodes'] for v in summaries.values())})
  finally:stop(server)
if __name__=='__main__':main()
