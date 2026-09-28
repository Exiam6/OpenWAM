import datetime,fcntl,json,subprocess
from pathlib import Path
from resume_state import inventory,BASE,S
R=BASE.parent
now=datetime.datetime.now().astimezone();counts,pending=inventory();record={'time':now.isoformat(),'inventory':counts}
def emit(action,**extra):
 record.update(action=action,**extra)
 (S/'monitor-status.json').write_text(json.dumps(record,indent=2)+'\n')
 with (S/'monitor-history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 if action not in ['running','waiting_gpu']:print(json.dumps(record),flush=True)
def main():
 with (S/'monitor.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  deadline=datetime.datetime.fromisoformat(json.loads((S/'protocol.json').read_text())['deadline'])
  if now>=deadline:
   subprocess.run(['systemctl','--user','stop','openwam-gpu5-worker.service'],check=False)
   subprocess.run(['systemctl','--user','disable','--now','openwam-gpu5-monitor.timer'],check=False)
   emit('deadline_reached');return
  active=subprocess.run(['systemctl','--user','is-active','--quiet','openwam-gpu5-worker.service']).returncode==0
  if active:emit('running');return
  if counts['duplicates']:emit('needs_attention_duplicate_records');return
  if not counts['pending']:
   emit('all_cells_attempted',complete=counts['completed']==1080)
   subprocess.run(['systemctl','--user','disable','--now','openwam-gpu5-monitor.timer'],check=False);return
  for p in [S/'paused.json',BASE/'paused.json',R/'migration-20260927/paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:
   if p.exists():emit('paused',marker=str(p));return
  previous=S/'status.json'
  if previous.exists():
   last=json.loads(previous.read_text())
   if last['phase']=='stopped':emit('needs_attention_systemic_failure',error=last.get('error'));return
  with (R/'gpu-5.lock').open('a') as gpu:
   try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:emit('waiting_gpu');return
  row=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
  if row[0]!='GPU-a4c607bb-f8f8-b999-19ea-0eb65290bb10' or int(row[1])>=100 or int(row[2])!=0:emit('waiting_gpu',gpu=row);return
  subprocess.run(['systemctl','--user','start','openwam-gpu5-worker.service'],check=True)
  emit('worker_started')
if __name__=='__main__':main()
