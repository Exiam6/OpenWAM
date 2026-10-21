"""Read-only independent selection; all100 rows required for official headline."""
import argparse,datetime as dt,json,pathlib,sys
R=pathlib.Path('/home/zifanz4/openwam-runtime');E=R/'ex8-e_rowrecovery_1021';D=R/'ex8-d_nativefull_1021'
sys.path.insert(0,str(D/'code'));from run_full import sha,score,classify_full,validate_full_suite

def audit():
 p=json.loads((E/'protocol.json').read_text());base=json.loads((D/'protocol.json').read_text());validate_full_suite(base)
 for name,digest in p['input_sha256'].items():assert sha(pathlib.Path(name))==digest,name
 selected=p['endpoint_selector'];assert len(selected)==100 and len({r['key'] for r in selected})==100
 assert {r['key'] for r in selected}=={r['key'] for r in base['rows']}
 complete=[];pending=[];failed=[]
 for expected,entry in zip(base['rows'],selected):
  assert entry['key']==expected['key']
  root=E if expected['key']=='demo_clean-adjust_bottle' else D
  f=root/'runs'/expected['key']/'result.json';assert str(f)==entry['result']
  if not f.exists():pending.append(entry['key']);continue
  r=json.loads(f.read_text())
  if r['status']=='failed':failed.append(dict(key=entry['key'],error=r.get('error'),partial_episodes=len(r['episodes'])));continue
  if r['status']!='complete':pending.append(entry['key']);continue
  assert r['protocol_sha256']==sha(root/'protocol.json')
  assert r['task']==expected['task'] and r['mode']==expected['mode']
  assert r['scored_episodes']==len(r['episodes'])==r['video_full_decode_passed']==100 and r['native_result_crosscheck']
  assert [e['episode_index'] for e in r['episodes']]==list(range(100))
  assert r['successes']==sum(e['success'] for e in r['episodes'])
  log=f.parent/'client.log';assert sha(log)==r['client_log_sha256'];text=log.read_text()
  assert score(text)==r['episodes'] and not classify_full(text,r['task'],True)['fatal']
  assert len(r['video_sha256'])==100
  actual=list((f.parent/'simulator/eval_result').rglob('episode*.mp4'));assert {str(x) for x in actual}==set(r['video_sha256'])
  for name,digest in r['video_sha256'].items():assert sha(pathlib.Path(name))==digest
  results=list((f.parent/'simulator/eval_result').rglob('_result.txt'));assert len(results)==1
  assert abs(float(results[0].read_text().strip().splitlines()[-1])-r['successes']/100)<1e-8
  complete.append(dict(key=entry['key'],task=r['task'],mode=r['mode'],episodes=100,successes=r['successes']))
 out=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),completed_rows=len(complete),scored_episodes=100*len(complete),pending=pending,failed=failed,excluded=p['excluded_attempts'],full_endpoint_complete=len(complete)==100,practical_agreement_passed=False,full_paper_reproduced=False,per_task=complete)
 if out['full_endpoint_complete']:
  rates={m:sum(x['successes'] for x in complete if x['mode']==m)/50 for m in ['demo_clean','demo_randomized']};rates['mean']=(rates['demo_clean']+rates['demo_randomized'])/2
  targets={'demo_clean':93.74,'demo_randomized':93.46,'mean':93.60};gaps={k:rates[k]-targets[k] for k in targets}
  out.update(rates_pct=rates,gaps_pp=gaps,practical_agreement_passed=all(abs(x)<=2 for x in gaps.values()),exact_published_numbers_matched=all(abs(x)<1e-8 for x in gaps.values()),conclusion='Complete endpoint; independent scientific gap interpretation remains required')
 return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path);v=a.parse_args();result=audit();s=json.dumps(result,indent=2)+'\n'
 if v.output:
  p=json.loads((E/'protocol.json').read_text());assert not any(pathlib.Path(x).exists() for x in p['pause_paths']);t=v.output.with_suffix('.pending');t.write_text(s);t.replace(v.output)
 print(s)
