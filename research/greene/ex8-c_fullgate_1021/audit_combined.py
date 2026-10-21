"""Independent CPU audit of actual attempts. Never edits their runtimes."""
import concurrent.futures,datetime as dt,json,math,pathlib,subprocess,time
from prepare import A,B,OUT,read,sha
from run_full import score,classify_full
PAUSE=pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')
DEST=OUT/'audit-20261021T1745Z'
def guard():
 assert not any(p.exists() for p in [PAUSE,A/'paused.json',B/'paused.json',OUT/'paused.json'])
def decode(path):
 guard();t=time.monotonic()
 p=subprocess.run([str(A/'bin/ffmpeg'),'-threads','1','-v','error','-i',str(path),'-threads','1','-f','null','-'],capture_output=True,timeout=90)
 assert p.returncode==0 and not p.stderr,(str(path),p.stderr.decode()[:400])
 return dict(path=str(path),sha256=sha(path),full_decode=True,decode_cpu_seconds=time.monotonic()-t)
def main():
 guard();DEST.mkdir(exist_ok=False)
 ap=read(A/'protocol.json');bp=read(B/'protocol.json');ar=read(A/'pilot/result.json');br=read(B/'pilot/result.json')
 assert ar['status']=='failed' and br['status']=='pilot_complete'
 assert bp['parent_protocol_sha256']==sha(A/'protocol.json') and bp['parent_result_sha256']==sha(A/'pilot/result.json')
 assert ap['checkpoint_sha256']==bp['checkpoint_sha256']
 sync=read(A.parent/'official-sync/latest.json');assert sync['passed'] and sync['after']==ap['official_sync']['after']==bp['official_sync']['after']
 inputs={**ap['input_sha256'],**bp['input_sha256']}
 for name,digest in inputs.items():guard();assert sha(pathlib.Path(name))==digest,name
 groups=[];videos=[];evidence={str(p):sha(p) for p in [A/'protocol.json',B/'protocol.json',A/'pilot/result.json',B/'pilot/result.json']}
 assert len(ar['groups'])==5 and len(br['groups'])==1
 for root,g in [(A,g) for g in ar['groups']]+[(B,g) for g in br['groups']]:
  guard();folder=root/'pilot'/f"{g['mode']}-{g['task']}";log=folder/'client.log';text=log.read_text();rows=score(text);classified=classify_full(text,g['task'],True)
  assert not classified['fatal'] and len(rows)==g['episodes']==5
  assert [r['scene_seed'] for r in rows]==g['accepted_scene_seeds']
  assert sum(r['success'] for r in rows)==g['successes']
  records=0;eps=set();record_index=[]
  for f in sorted((folder/'policy-inputs').glob('ep*/step_*/meta.json')):
   m=read(f);records+=1;eps.add(int(f.parent.parent.name[2:]));record_index.append((str(f),sha(f)))
   for k in ['action','state']:
    assert len(m[k])==20 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in m[k]),str(f)
   for camera in ['head','left','right']:assert (f.parent/(camera+'.png')).stat().st_size>64
  assert records==g['debug_records'] and eps==set(range(5))
  assert len(g['videos'])==5 and {pathlib.Path(v).name for v in g['videos']}=={f'episode{i}.mp4' for i in range(5)}
  videos.extend(map(pathlib.Path,g['videos']));evidence[str(log)]=sha(log)
  groups.append(dict(task=g['task'],mode=g['mode'],attempt=str(root),episodes=rows,successes=g['successes'],seconds=g['seconds'],finite_20d_records=records,camera_files_present=records*3,rejected_candidate_exceptions=classified['rejected_candidates']))
  (DEST/f"{g['mode']}-{g['task']}-record-index.json").write_text(json.dumps(record_index)+'\n')
 assert len({(g['task'],g['mode']) for g in groups})==6
 partial=score((A/'pilot/demo_randomized-place_object_basket/client.log').read_text());assert len(partial)==2
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:decoded=list(pool.map(decode,videos))
 total=sum(g['seconds'] for g in groups);nominal=total/30*10000/3600+5
 result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),kind='INDEPENDENT_CPU_READINESS_AUDIT',groups=groups,episodes=30,successes=sum(g['successes'] for g in groups),finite_20d_records=sum(g['finite_20d_records'] for g in groups),decoded_videos=decoded,immutable_input_files_verified=len(inputs),original_failed_attempt_retained=True,original_partial_episodes_excluded=2,original_partial_scene_seeds=[r['scene_seed'] for r in partial],all_pilot_episodes_excluded_from_full_score=True,mechanical_checks_passed=True,visual_review_pending=True,combined_readiness_passed=False,full_cost_estimate_gpu_hours=nominal,conservative_startup_and_guard_estimate_gpu_hours=nominal-5+100*(600+120)/3600,cost_note='Pilot extrapolation, not measured full-suite cost; 94 task/mode rows unmeasured. Nominal adds 5 startup GPUh; conservative estimate adds 600s startup and 120s guard per job. Video decode time is measured separately below.',authorized_ceiling_gpu_hours=200,cost_gate_passed=False,official_compatibility='Tracking pull unchanged at '+sync['after']+'; frozen sources unchanged',evidence_sha256=evidence,full_endpoint_complete=False,full_paper_reproduced=False)
 guard();(DEST/'combined-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['groups','decoded_videos','evidence_sha256']},indent=2))
if __name__=='__main__':main()
