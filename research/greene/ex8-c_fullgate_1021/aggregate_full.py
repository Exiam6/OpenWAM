"""Complete native endpoint only; no partial-suite headline or selective exclusions."""
import argparse,datetime as dt,json,pathlib,random,time
from run_full import validate_full_suite,score,classify_full,sha
ROOT=pathlib.Path('/home/zifanz4/openwam-runtime/ex8-c_nativefull_1021')
def read(p):return json.loads(p.read_text())
def main():
 assert not pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
 p=read(ROOT/'protocol.json');validate_full_suite(p);complete=[];failed=[];pending=[]
 for expected in p['rows']:
  f=ROOT/'runs'/expected['key']/'result.json'
  if not f.exists():pending.append(expected['key']);continue
  r=read(f)
  if r['status']=='failed':failed.append(dict(key=expected['key'],error=r.get('error'),partial_episodes=len(r['episodes'])));continue
  if r['status']!='complete':pending.append(expected['key']);continue
  import hashlib
  assert r['protocol_sha256']==hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
  assert all(b['scene_seed']>a['scene_seed'] for a,b in zip(r['episodes'],r['episodes'][1:]))
  assert r['task']==expected['task'] and r['mode']==expected['mode']
  assert r['scored_episodes']==len(r['episodes'])==100 and r['video_full_decode_passed']==100 and r['native_result_crosscheck']
  assert [e['episode_index'] for e in r['episodes']]==list(range(100))
  assert r['successes']==sum(e['success'] for e in r['episodes'])
  log=f.parent/'client.log';assert sha(log)==r['client_log_sha256']
  text=log.read_text();assert score(text)==r['episodes'] and not classify_full(text,r['task'],True)['fatal']
  assert len(r['video_sha256'])==100
  for name,digest in r['video_sha256'].items():assert sha(pathlib.Path(name))==digest

  complete.append(dict(task=r['task'],mode=r['mode'],episodes=100,successes=r['successes'],rate_pct=r['successes']))
 result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),completed_task_mode_rows=len(complete),total_task_mode_rows=100,scored_episodes=len(complete)*100,per_task=complete,pending=pending,failed=failed,full_endpoint_complete=len(complete)==100,practical_agreement_passed=False,exact_published_numbers_matched=False,pilot_excluded=True)
 if len(complete)==100:
  clean={r['task']:r['rate_pct'] for r in complete if r['mode']=='demo_clean'};rand={r['task']:r['rate_pct'] for r in complete if r['mode']=='demo_randomized'};assert clean.keys()==rand.keys() and len(clean)==50
  c=sum(clean.values())/50;r=sum(rand.values())/50;rates=dict(clean=c,randomized=r,mean=(c+r)/2);targets=dict(clean=93.74,randomized=93.46,mean=93.6);gaps={k:rates[k]-targets[k] for k in rates}
  rng=random.Random(20261021);tasks=sorted(clean);samples={k:[] for k in rates}
  for _ in range(10000):
   draw=rng.choices(tasks,k=50);a=sum(clean[t] for t in draw)/50;b=sum(rand[t] for t in draw)/50
   for k,v in [('clean',a),('randomized',b),('mean',(a+b)/2)]:samples[k].append(v)
  ci={k:[sorted(v)[249],sorted(v)[9749]] for k,v in samples.items()}
  result.update(rates_pct=rates,targets_pct=targets,gaps_pp=gaps,task_bootstrap_95pct=ci,task_bootstrap_note='Paired task resampling, 10000 replicates. Describes across-task variation; paper seed variance is unavailable.',practical_agreement_passed=all(abs(v)<=2 for v in gaps.values()),exact_published_numbers_matched=all(abs(v)<1e-8 for v in gaps.values()),local_tolerance_pp=2,software_difference='Benchmark torch 2.7.0+cu128 vs documented 2.4.1')
 assert not pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
 t=ROOT/'readout.tmp';t.write_text(json.dumps(result,indent=2)+'\n');t.replace(ROOT/'readout.json');return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--watch',action='store_true');args=a.parse_args()
 while True:
  result=main();print(json.dumps({k:result[k] for k in ['updated','completed_task_mode_rows','full_endpoint_complete','practical_agreement_passed']}),flush=True)
  if not args.watch or result['full_endpoint_complete'] or dt.datetime.now(dt.timezone.utc)>dt.datetime.fromisoformat(read(ROOT/'protocol.json')['deadline']):break
  time.sleep(30)
