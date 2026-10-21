"""Bounded CPU continuation: audit the pilot, then fail-closed L4 registration."""
import datetime as dt,json,pathlib,subprocess,time,traceback
from audit_native import audit,ROOT,read,write
SOURCE=pathlib.Path(__file__).resolve().parent
if __name__=='__main__':
 out=ROOT/'independent-audit/continuation.json'
 try:
  while True:
   now=dt.datetime.now(dt.timezone.utc);r=audit()
   state=dict(updated=now.isoformat(),pilot_status=r['pilot_status'],scored_episodes=r['pilot_scored_episodes'],next_gate=r['next_gate'],full_paper_reproduced=False)
   write(out,state);print(json.dumps(state),flush=True)
   if r['pilot_status']=='failed':break
   if now>=dt.datetime.fromisoformat(read(ROOT/'protocol.json')['deadline']):
    state['next_gate']='pilot_deadline_reached';write(out,state);break
   if r['pilot_status']=='pilot_complete':
    if not r['within_registered_200_gpu_hour_planning_ceiling']:
     state.update(status='held_for_cost_review',full_estimate_gpu_hours=r['full_planning_estimate_gpu_hours'],registered_ceiling_gpu_hours=200);write(out,state);break
    with (ROOT/'independent-audit/full-registration.log').open('x') as log:
     done=subprocess.run(['/usr/bin/python3',str(SOURCE/'register_full.py')],stdout=log,stderr=subprocess.STDOUT,timeout=180)
    state.update(status='L4_registered' if done.returncode==0 else 'registration_failed',exit_code=done.returncode);write(out,state);break
   time.sleep(30)
 except BaseException as e:
  write(out,dict(status='continuation_failed',error=repr(e),full_paper_reproduced=False,updated=dt.datetime.now(dt.timezone.utc).isoformat()));raise
