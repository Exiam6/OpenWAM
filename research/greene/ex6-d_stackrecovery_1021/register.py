import pathlib,json,hashlib,shutil,datetime as dt,subprocess
R=pathlib.Path('/home/zifanz4/openwam-runtime');N=R/'nightly-gpu';C=R/'ex6-c_resourcerecovery_1020';D=R/'ex6-d_stackrecovery_1021';S=pathlib.Path('/home/zifanz4/openwam-vae-control/research/greene/ex6-d_stackrecovery_1021')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
p=read(C/'protocol.json');state=read(C/'transfer-status.json');task='stack_blocks_two';assert sum(x['missing_or_unverified'] for x in state['tasks'][task]['arms'].values())==9
assert not any(x['status']=='complete' for x in [read(C/'tasks'/task/'result.json')])
charged=0
for pattern in ['ex6a-*.json','ex6c-*.json']:
 for f in (N/'claims').glob(pattern):
  claim=read(f);assert claim['status'] in ['complete','failed'];charged+=claim['worker_result']['occupied_parent_seconds']
assert charged+14400<=p['total_gpu_hour_cap']*3600
sync=read(R/'official-sync/latest.json');assert sync['passed']; changed=subprocess.check_output(['git','-C','/home/zifanz4/openwam-official-main','diff','--name-only',p['official_sync']['after'],sync['after']],text=True).splitlines(); assert changed==['assets/repo_images/WeChat_Group.png']; p['official_compatibility']='Only official WeChat_Group.png changed; frozen scientific code, weights and evaluation remain unchanged.'
D.mkdir(exist_ok=False)
for folder in ['template','robot-configs']:shutil.copytree(C/folder,D/folder)
shutil.copyfile(C/'base-environment.json',D/'base-environment.json')
for name in ['run_task.py','PLAN.md']:shutil.copyfile(S/name,D/name)
write(D/'prior-transfer-status.json',state)
p.update(name=D.name,tasks={task:p['tasks'][task]},missing={task:p['missing'][task]},created=dt.datetime.now(dt.timezone.utc).isoformat(),official_sync=sync,source_commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip(),previously_charged_seconds=charged,recovery_reserved_seconds=14400,combined_worst_case_gpu_hours=(charged+14400)/3600,recovery_rule='One additional missing-only bounded recovery after the earlier recovery was resource-interrupted. Preserve both earlier attempts.')
assert sum(len(v) for v in p['missing'][task].values())==9
for arm in p['arms']:
 expected={x['scene_seed'] for x in state['tasks'][task]['arms'][arm['name']]['rows'] if x['status']=='verified'}
 assert not expected.intersection(p['missing'][task][arm['name']])
for f in D.rglob('*'):
 if f.is_file():p['code_sha256'][str(f)]=sha(f)
p['recovery_inputs_sha256'][str(N/'claims/ex6c-stack-blocks-two.json')]=sha(N/'claims/ex6c-stack-blocks-two.json')
write(D/'protocol.json',p)
argv=['/usr/bin/python3',str(D/'run_task.py'),'--task',task]
subprocess.run(argv+['--check-admission'],check=True)
job=dict(id='ex6d-stack-blocks-two',status='ready',scientific_identity=f'{D.name}/missing-only/stack_blocks_two/seed42',max_seconds=14400,authorization_evidence=str(D/'protocol.json'),authorization_sha256=sha(D/'protocol.json'),pause_paths=[str(D/'paused.json'),str(C/'paused.json'),str(R/'ex6-a_tasktransfer_1020/paused.json'),'/home/zifanz4/.local/state/openwam-selfcheck/paused.json'],host_commands={'gpu01':dict(argv=argv,admission_argv=argv+['--check-admission'],cwd=str(D),input_sha256={str(D/'protocol.json'):sha(D/'protocol.json'),str(D/'run_task.py'):sha(D/'run_task.py')},outputs_absent=[str(D/'tasks'/task)])})
write(D/'registry.json',dict(job=job,protocol_sha256=sha(D/'protocol.json'),missing_episodes=9,registered_before_launch=True))
dest=N/'jobs/ex6d-stack-blocks-two.json';assert not dest.exists();tmp=dest.with_suffix('.pending');write(tmp,job);tmp.replace(dest)
print(json.dumps(dict(missing=9,combined_worst_case_gpu_hours=p['combined_worst_case_gpu_hours'],job=job['id'],protocol_sha256=sha(D/'protocol.json'))))
