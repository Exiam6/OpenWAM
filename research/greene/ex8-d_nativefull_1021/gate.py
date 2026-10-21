"""CPU scaling gate; release remaining already-registered rows after native evidence."""
import datetime as dt,json,pathlib,subprocess,time
from run_full import FULL,ROOT,read,sha,classify_full
N=FULL.parent/'nightly-gpu';PAUSE=pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')
def write(p,v):
 t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def once():
 p=read(FULL/'protocol.json');registry=read(FULL/'registry.json');jobs=registry['jobs'];first=jobs[0];out=pathlib.Path(first['host_commands']['gpu01']['outputs_absent'][0]);status=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),status='waiting_for_native_first_rollout',submitted_rows=1,full_endpoint_complete=False)
 assert sha(FULL/'protocol.json')==registry['protocol_sha256']
 assert sha(FULL/'registry.json')==(FULL/'registry.sha256').read_text().strip()
 assert len(jobs)==len(p['rows'])==100
 for job,row in zip(jobs,p['rows']):
  assert job['authorization_sha256']==registry['protocol_sha256'] and job['max_seconds']==row['max_seconds']
  assert job['host_commands']['gpu01']['argv']==['/usr/bin/python3',str(FULL/'code/run_full.py'),'--key',row['key']]
 for name,digest in p['input_sha256'].items():assert sha(pathlib.Path(name))==digest,name
 assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
 if any(pathlib.Path(x).exists() for x in p['pause_paths']):status['status']='paused';return status
 state=read(out/'result.json') if (out/'result.json').exists() else {}
 claim=read(N/'claims'/f'{first["id"]}.json') if (N/'claims'/f'{first["id"]}.json').exists() else {}
 if state.get('status')=='failed' or claim.get('status')=='failed':status.update(status='hold_first_row_failed',error=state.get('error') or claim.get('error'));return status
 rows=state.get('episodes',[])
 if not rows:return status
 assert [r['episode_index'] for r in rows]==list(range(len(rows)))
 if not any(r['success'] for r in rows[:3]):
  if len(rows)>=3:status['status']='hold_first_three_native_rollouts_unsuccessful'
  return status
 assert state.get('server_ready')
 text=(out/'client.log').read_text();assert not classify_full(text,p['rows'][0]['task'])['fatal']
 videos=list((out/'simulator/eval_result').rglob('episode0.mp4'));assert len(videos)==1
 subprocess.run([str(ROOT/'bin/ffmpeg'),'-v','error','-i',str(videos[0]),'-f','null','-'],check=True,timeout=60,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 for job in jobs[1:]:
  if any(pathlib.Path(x).exists() for x in p['pause_paths']):raise RuntimeError('Paused during bounded release')
  dest=N/'jobs'/f'{job["id"]}.json';claim=N/'claims'/f'{job["id"]}.json'
  if dest.exists():assert read(dest)==job
  else:assert not claim.exists();write(dest,job)
 status.update(status='remaining_rows_released',submitted_rows=100,first_observed_episodes=len(rows),first_observed_successes=sum(r['success'] for r in rows),first_video_sha256=sha(videos[0]));return status
if __name__=='__main__':
 while True:
  try:s=once()
  except BaseException as e:write(FULL/'stage-state.json',dict(status='gate_failed',error=repr(e),updated=dt.datetime.now(dt.timezone.utc).isoformat()));raise
  write(FULL/'stage-state.json',s);print(json.dumps(s),flush=True)
  if s['status']=='remaining_rows_released' or s['status'].startswith('hold_'):break
  time.sleep(10)
