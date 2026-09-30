"""One endtoend checkpoint on one L40S: frozen-hash check, server, development gate,
then (only with a frozen protocol.json) the requested task x condition cells.

--gate-only runs the training-scene (seed 10) preflight and a full development
rollout, and never touches a heldout scene. Finite; failures retained, no retry.
"""
import argparse,datetime,hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
REPO=Path('/scratch/zz4330/OpenWAM');B=REPO/'research/greene/endtoend-eval';SCRIPT=B/'scripts'
RUNTIME=Path('/scratch/zz4330/openwam-runtime');ROOT=RUNTIME/'endtoend-root';E=ROOT/'endtoend-20260923'
POLICY_PY='/scratch/zz4330/conda/envs/openwam-policy/bin/python';BENCH_PY='/scratch/zz4330/conda/envs/openwam-bench/bin/python'
VIEW=json.loads((REPO/'research/greene/records/endtoend-view.json').read_text())['checkpoints']
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
def child(cmd,env,log,seconds,cwd):
 with log.open('x') as f:
  proc=subprocess.Popen(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
  try:
   while proc.poll() is None:
    assert time.monotonic()-start<seconds,cmd;time.sleep(5)
   assert proc.returncode==0,(proc.returncode,str(log))
  finally:stop(proc)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--route',required=True,choices=['wan','svae','pca']);ap.add_argument('--seed',type=int,required=True,choices=[42,43,44])
 ap.add_argument('--gate-only',action='store_true');ap.add_argument('--cells',default='[]',help='JSON list of [task, condition]');a=ap.parse_args()
 out=Path(os.environ['OPENWAM_RUN_DIR']);run=f'{a.route}-seed{a.seed}';v=VIEW[run];ckpt=Path(v['view'])/'checkpoint_step_12000.safetensors'
 cells=json.loads(a.cells);assert a.gate_only!=bool(cells),'give --gate-only or --cells'
 protocol=None
 if cells:
  protocol=json.loads((B/'protocol.json').read_text());assert protocol.get('frozen') is True,'protocol not frozen'
  assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(protocol['deadline']),'past frozen deadline'
  for f,want in protocol['code_sha256'].items():assert sha(f)==want,f
 t=time.monotonic();got=sha(ckpt);assert got==v['checkpoint_sha256'],(ckpt,got);assert sha(Path(v['view'])/'config.yaml')==v['view_config_sha256']
 code={str(p):sha(p) for p in sorted(SCRIPT.glob('*.py'))}
 write(out/'run-record.json',{'time':now(),'host':socket.gethostname(),'slurm_job':os.environ.get('SLURM_JOB_ID'),'route':a.route,'seed':a.seed,'checkpoint':str(ckpt),'checkpoint_sha256':got,'checkpoint_hash_seconds':time.monotonic()-t,'gate_only':a.gate_only,'cells':cells,'code_sha256':code,'protocol':protocol})
 if cells:child([BENCH_PY,'-u',str(SCRIPT/'verify_cohort.py'),'--output',str(out/'cohort-integrity.json')],dict(os.environ),out/'cohort-integrity.log',1800,REPO)
 pci=subprocess.check_output(['nvidia-smi','--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip().splitlines();assert len(pci)==1,pci
 port=19400+int(os.environ.get('SLURM_JOB_ID','0'))%500;probe=socket.socket();probe.bind(('127.0.0.1',port));probe.close()
 env=dict(os.environ,PYTHONPATH=str(E/'OpenWAM'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
 sim=dict(env,EXPECTED_RENDER_PCI=pci[0],WAM_ROOT=str(ROOT),ROBOTWIN_PATH=str(RUNTIME/'benchmarks/RoboTwin'),ROBOTWIN_RUNTIME_ROOT=str(out/'runtime'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1');sim.pop('PYTHONPATH',None)
 with (out/'server.log').open('x') as f:
  t=time.monotonic()
  server=subprocess.Popen([POLICY_PY,'-u',str(SCRIPT/'serve_policy.py'),'--checkpoint',str(ckpt),'--port',str(port),'--log',str(out/'episode-rng.jsonl')],env=env,cwd=E/'OpenWAM',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while 'ENDTOEND_SERVER_READY' not in (out/'server.log').read_text(errors='replace'):
    assert server.poll() is None,'server exited before ready';assert time.monotonic()-t<1800,'server not ready in 1800 s';time.sleep(5)
   startup=time.monotonic()-t;smoke=out/'development';smoke.mkdir(exist_ok=False);t=time.monotonic()
   child([BENCH_PY,'-u',str(SCRIPT/'simulation_preflight.py'),'--port',str(port),'--output',str(smoke)],sim,smoke/'run.log',3600,REPO)
   result=json.loads((smoke/'result.json').read_text());dev=result['tasks'][0]['full_development_rollout']
   assert result['passed'] and not result['heldout_scenes_used'] and dev['completed']
   gate={'passed':True,'time':now(),'route':a.route,'seed':a.seed,'checkpoint_sha256':got,'server_startup_seconds':startup,'preflight_seconds':time.monotonic()-t,'development_rollout':dev,'development_result':str(smoke/'result.json'),'code_sha256':code}
   write(out/f'simulation-preflight-{a.route}-{a.seed}.json',gate);print('GATE',json.dumps(gate),flush=True)
   summaries={}
   for task,condition in cells:
    assert condition in protocol['evaluation']['conditions'] and task in protocol['evaluation']['tasks']
    manifest=json.loads((E/'scenes'/task/'manifest.json').read_text());seeds=[int(x['seed']) for x in manifest['accepted']][:protocol['evaluation']['scenes_per_task']]
    dest=out/task/condition;dest.mkdir(parents=True,exist_ok=False)
    child([BENCH_PY,'-u',str(SCRIPT/'evaluate_policy.py'),'--scene-seeds',json.dumps(seeds),'--task',task,'--route',a.route,'--seed',str(a.seed),'--condition',condition,'--port',str(port),'--output',str(dest)],sim,dest/'run.log',len(seeds)*660,REPO)
    s=json.loads((dest/'summary.json').read_text());summaries[f'{task}/{condition}']={k:s[k] for k in ['episodes','successes','technical_failures','complete']};print(task,condition,summaries[f'{task}/{condition}'],flush=True)
   write(out/'complete.json',{'time':now(),'gate_only':a.gate_only,'cells':summaries})
  finally:stop(server)
if __name__=='__main__':main()
