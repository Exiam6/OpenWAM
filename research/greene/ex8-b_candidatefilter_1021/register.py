import pathlib,json,hashlib,shutil,subprocess,datetime as dt
R=pathlib.Path('/home/zifanz4/openwam-runtime');BASE=R/'ex8-a_officialrepro_1021';ROOT=R/'ex8-b_candidatefilter_1021';SOURCE=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert not ROOT.exists()
sync=read(R/'official-sync/latest.json');assert sync['passed'] and (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(sync['time'])).total_seconds()<3600
p=read(BASE/'protocol.json');assert sync['after']==p['official_sync']['after'];old=read(BASE/'pilot/result.json');assert old['status']=='failed' and len(old['groups'])==5
assert sum(g['episodes'] for g in old['groups'])==25 and old['occupied_seconds']+3600<10800
ROOT.mkdir()
for name in ['official','RoboTwin','bin']:(ROOT/name).symlink_to(BASE/name,target_is_directory=True)
for name in ['run_native.py','classify_log.py','register.py','PLAN.md','audit_recovery.py']:shutil.copyfile(SOURCE/name,ROOT/name);p['input_sha256'][str(ROOT/name)]=sha(ROOT/name)
p.update(name=ROOT.name,created=dt.datetime.now(dt.timezone.utc).isoformat(),deadline='2026-10-23T15:00:00+00:00',source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),official_sync=sync,pilot_tasks=['place_object_basket'],pilot_modes=['demo_randomized'],max_seconds=3600,pilot_gpu_hour_cap=1,parent_protocol_sha256=sha(BASE/'protocol.json'),parent_result_sha256=sha(BASE/'pilot/result.json'),recovery_scope='One complete five-episode missing group; excludes original two partial duplicate episodes',original_attempts_unchanged=True,full_registered_for_dispatch=False)
write(ROOT/'protocol.json',p)
argv=['/usr/bin/python3',str(ROOT/'run_native.py')];subprocess.run(argv+['--check-admission'],check=True)
job=dict(id='ex8b-native-candidate-recovery',status='ready',scientific_identity='ex8-b_candidatefilter_1021/randomized-place-object-basket/seed0',max_seconds=3600,authorization_evidence=str(ROOT/'protocol.json'),authorization_sha256=sha(ROOT/'protocol.json'),pause_paths=[str(ROOT/'paused.json'),'/home/zifanz4/.local/state/openwam-selfcheck/paused.json'],host_commands={'gpu01':dict(argv=argv,admission_argv=argv+['--check-admission'],cwd=str(ROOT),input_sha256={str(ROOT/'protocol.json'):sha(ROOT/'protocol.json')},outputs_absent=[str(ROOT/'pilot')])})
write(ROOT/'registry.json',dict(stage='registered_L1_L3_recovery',parent_failure_retained=True,job=job,protocol_sha256=sha(ROOT/'protocol.json')))
target=R/'nightly-gpu/jobs'/f'{job["id"]}.json';assert not target.exists();temp=target.with_suffix('.pending');write(temp,job);temp.replace(target)
print(json.dumps(dict(registered=job['id'],protocol_sha256=sha(ROOT/'protocol.json'),max_gpu_hours=1,parent_plus_reserved_hours=(old['occupied_seconds']+3600)/3600)))
