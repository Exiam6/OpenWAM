"""New fixed cohort; reuse unchanged native expert worker with provenance checks."""
import argparse,datetime,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ['WAM_ROOT']);O=ROOT/'results/goalaux-20260922';RUN=O/'fresh';TASKS=['adjust_bottle','handover_block','place_object_basket']

def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pci(value):
 domain,bus,last=value.split(':');device,function=last.split('.');return tuple(int(x,16) for x in [domain,bus,device,function])

def worker(task,seed):
 ref=ROOT/'scripts/goalaux/collect_reference.py';spec=importlib.util.spec_from_file_location('native_collection_reference',ref);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native);native.ROOT=ROOT;native.RUN=RUN
 sys.path.insert(0,str(ROOT/'OpenWAM/benchmarks/robotwin'));import eval_policy_wrapper as wrapper
 original_bootstrap=wrapper.bootstrap_robotwin_module;provenance=[];dest=RUN/task/f'seed-{seed}'
 def checked_bootstrap(*args,**kwargs):
  mod=original_bootstrap(*args,**kwargs);factory=mod.class_decorator
  def checked_factory(task_name):
   env=factory(task_name);setup=env.setup_demo
   def checked_setup(*aa,**kw):
    value=setup(*aa,**kw)
    device=env.scene.render_system.device
    planners={side:type(getattr(env.robot,side+'_planner',None)).__name__ for side in ['left','right']}
    record={'need_plan':kw.get('need_plan'),'renderer_name':device.name,'renderer_cuda_id':device.cuda_id,'renderer_pci':device.pci_string,'planners':planners}
    provenance.append(record);write(dest/'runtime-provenance.json',provenance)
    assert pci(device.pci_string)==pci(os.environ['EXPECTED_RENDER_PCI']),'Renderer used unexpected GPU'
    if kw.get('need_plan'):assert all(name=='CuroboPlanner' for name in planners.values()),'Planner backend mismatch; stop before actions'
    return value
   env.setup_demo=checked_setup;return env
  mod.class_decorator=checked_factory;return mod
 wrapper.bootstrap_robotwin_module=checked_bootstrap
 native.worker(task,seed)

def parent():
 plan=json.loads((O/'matched-protocol.json').read_text());implementation=json.loads((RUN/'implementation-freeze.json').read_text());launch=json.loads((RUN/'launch.json').read_text());deadline=datetime.datetime.fromisoformat(launch['collection_deadline'])
 assert not (RUN/'manifest.json').exists() and not (RUN/'started.json').exists(),'No cohort replay'
 for path,expected in implementation['sha256'].items():assert sha(ROOT/path)==expected,path
 assert json.loads((RUN/'seed-exclusion.json').read_text())['passed']
 assert json.loads((O/'native-preflight/result.json').read_text())['passed']
 write(RUN/'started.json',{'started_at':datetime.datetime.now().astimezone().isoformat(),'collection_deadline':deadline.isoformat(),'study_deadline':plan['deadline']});start=time.monotonic();attempts=[];accepted={task:[] for task in TASKS}
 for task in TASKS:
  for offset in range(40):
   for p in [O/'paused.json',ROOT/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),p
   remaining=(deadline-datetime.datetime.now().astimezone()).total_seconds();assert remaining>0,'Original collection deadline reached'
   seed=plan['new_confirmation_seeds'][task]+offset;dest=RUN/task/f'seed-{seed}';assert not dest.exists(),'No attempted-scene replay';dest.mkdir(parents=True)
   with (dest/'worker.log').open('w') as log:
    process=subprocess.Popen([sys.executable,'-u',__file__,'--task',task,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(dest/'owned-worker.json',{'pid':process.pid,'process_group':process.pid,'task':task,'seed':seed})
    try:code=process.wait(timeout=min(600,remaining))
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
     code=124
   path=dest/'result.json';res=json.loads(path.read_text()) if path.exists() else {'task':task,'seed':seed,'accepted':False,'outcome':'infrastructure_failure','error':'worker did not produce result'};res['exit_code']=code;attempts.append(res)
   if res.get('accepted'):accepted[task].append(res)
   write(RUN/'progress.json',{'attempts':attempts,'accepted_counts':{k:len(v) for k,v in accepted.items()},'elapsed_seconds':time.monotonic()-start,'collection_deadline':deadline.isoformat()});print(task,seed,res.get('outcome'),{k:len(v) for k,v in accepted.items()},flush=True)
   if code or res.get('outcome')=='infrastructure_failure':raise RuntimeError(f'Infrastructure failure retained: {task} {seed}; no retry')
   if len(accepted[task])==20:break
 write(RUN/'manifest.json',{'protocol':'first20expert-feasible scenes within40consecutive seeds per task','attempts':attempts,'accepted':accepted,'complete':all(len(v)==20 for v in accepted.values()),'no_learned_policy_scoring':True,'elapsed_seconds':time.monotonic()-start,'collection_deadline':deadline.isoformat()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS);p.add_argument('--seed',type=int);a=p.parse_args()
 if a.task:worker(a.task,a.seed)
 else:parent()
