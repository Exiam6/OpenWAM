#!/usr/bin/env python3
"""Snapshot a live oidn-outcome run (per-scene results + log tails) into records/ and commit + push if it changed.
   python3 update_oidn_outcome_progress.py --job <slurm job id> [--commit]"""
import argparse,datetime,glob,json,os,subprocess
from pathlib import Path
REPO=Path('/scratch/zz4330/OpenWAM');RT=Path('/scratch/zz4330/openwam-runtime')
a=argparse.ArgumentParser();a.add_argument('--job',required=True);a.add_argument('--commit',action='store_true');a=a.parse_args()
run=RT/f'runs/oidn-outcome-{a.job}'
def tail(p,n=15):
    try: return Path(p).read_text(errors='replace').splitlines()[-n:]
    except OSError: return []
state=subprocess.run(['sacct','-X','-n','-j',a.job,'-o','State,Elapsed,NodeList'],capture_output=True,text=True).stdout.split()
tasks={}
for d in sorted(glob.glob(str(run/'*/clean'))):
    task=Path(d).parent.name;scenes=[]
    for f in sorted(glob.glob(d+'/seed-*/result.json')):
        x=json.load(open(f));scenes.append({'scene_seed':x['scene_seed'],'success':x['success'],'steps':x['steps'],
          'initial_render_mae_max':round(max(x['initial_render_mae'].values()),4),'action_seed':x['action_seed']})
    tasks[task]={'completed':len(scenes),'successes':sum(s['success'] for s in scenes),'scenes':scenes,'run_log_tail':tail(d+'/run.log',8)}
rec={'record':f'oidn-outcome-{a.job}-progress','task':'oidn-outcome-20261004 follow-up (handover_block + place_object_basket, OIDN 2.3.3)',
     'job':int(a.job),'slurm':state,'tasks':tasks,'exit':json.load(open(run/'exit.json')) if (run/'exit.json').exists() else None,
     'slurm_log_tail':tail(RT/f'logs/owam-oidn-outcome.{a.job}.out'),
     'compare':{'18771822_oidn_2.0.1':{'handover_block':'20/20','place_object_basket':'18/20'},'shenlong_oidn_2.3.3':{'handover_block':'1/20','place_object_basket':'0/19'}}}
out=REPO/f'research/greene/records/oidn-outcome-{a.job}-progress.json'
old=json.loads(out.read_text()) if out.exists() else None
if old is not None: old.pop('updated',None)
if old==rec: print('no change');raise SystemExit
rec['updated']=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
out.write_text(json.dumps(rec,indent=1,ensure_ascii=False)+'\n')
summary=', '.join(f"{t} {v['successes']}/{v['completed']}" for t,v in tasks.items()) or 'starting'
print(summary)
if a.commit:
    g=lambda *c:subprocess.run(['git','-C',str(REPO),*c],capture_output=True,text=True)
    g('add',str(out))
    if g('diff','--cached','--quiet').returncode:
        g('commit','-m',f'research/greene: oidn-outcome {a.job} progress - {summary}\n\nCo-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>')
        p=g('push','git@github.com:Exiam6/OpenWAM.git','HEAD:research/progress-20260927');print('push',p.returncode,p.stderr[-300:])
