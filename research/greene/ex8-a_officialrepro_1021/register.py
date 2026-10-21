import pathlib,json,hashlib,shutil,subprocess,datetime as dt,re,concurrent.futures
R=pathlib.Path('/home/zifanz4/openwam-runtime');ROOT=R/'ex8-a_officialrepro_1021';SOURCE=pathlib.Path(__file__).parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert not (ROOT/'protocol.json').exists()
sync=read(R/'official-sync/latest.json');assert sync['passed'];assert (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(sync['time'])).total_seconds()<3600
for name in ['run_native.py','PLAN.md','register.py']:shutil.copyfile(SOURCE/name,ROOT/name)
inputs={}
for tree in ['official','RoboTwin']:
 base=ROOT/tree
 tracked=subprocess.check_output(['git','-C',str(base),'ls-files','-z']).decode().split('\0')
 for rel in tracked:
  p=base/rel
  if rel and p.is_file():inputs[str(p)]=sha(p)
for name in ['run_native.py','PLAN.md','register.py','asset-relocation.json']:inputs[str(ROOT/name)]=sha(ROOT/name)
ckpt=R/'ex4-h_officialcalibration_1017/checkpoint';expected=next(a for a in read(R/'ex6-a_tasktransfer_1020/protocol.json')['arms'] if a['name']=='official_reference')
for f in ckpt.rglob('*'):
 if f.is_file() and f.suffix!='.safetensors':inputs[str(f)]=sha(f)
for f in (ROOT/'RoboTwin/assets/embodiments/aloha-agilex').glob('*.yml'):inputs[str(f)]=sha(f)
# Original cuRobo files are byte-identical to RoboTwin's official bundled source.
a=R/'curobo-src/src/curobo';b=ROOT/'RoboTwin/envs/curobo/src/curobo';matched=0
for f in a.rglob('*'):
 if f.is_file() and f.suffix in ['.py','.cu','.h','.hpp','.cpp']:
  peer=b/f.relative_to(a);assert peer.is_file() and sha(f)==sha(peer),str(f);matched+=1
# Freeze the downloaded native background corpus; exclude unused training textures from per-launch checks.
textures=sorted((ROOT/'RoboTwin/assets/background_texture/unseen').glob('*.png'));assert len(textures)>0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:tex=dict(zip(map(str,textures),pool.map(sha,textures)))
write(ROOT/'texture-manifest.json',dict(source='https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/background_texture.zip',download_sha256=sha(ROOT/'background_texture.zip'),files=tex))
inputs.update(tex);inputs[str(ROOT/'texture-manifest.json')]=sha(ROOT/'texture-manifest.json')
alltasks=re.search(r'ROBOTWIN_ALL_TASKS=\((.*?)\)',(ROOT/'official/benchmarks/robotwin/multi_eval.sh').read_text(),re.S).group(1).split();assert len(alltasks)==50
versions=json.loads(subprocess.check_output([str(R/'benchmark-env/bin/python'),'-c','import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in ["torch","sapien","warp-lang","mplib"]}))'],text=True))
p=dict(name=ROOT.name,created=dt.datetime.now(dt.timezone.utc).isoformat(),deadline='2026-10-22T15:00:00+00:00',source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),official_sync=sync,checkpoint=expected['checkpoint'],checkpoint_sha256=expected['sha256'],input_sha256=inputs,pilot_tasks=['adjust_bottle','handover_block','place_object_basket'],pilot_episodes_per_group=5,max_seconds=10800,pilot_gpu_hour_cap=3,full_tasks=alltasks,full_episodes_per_task_per_mode=100,full_modes=['demo_clean','demo_randomized'],full_planned_episodes=10000,full_registered_for_dispatch=False,benchmark_versions=versions,compatibility_deviations=['Benchmark torch 2.7.0+cu128 vs paper verified 2.4.1; Blackwell hardware. No algorithmic fallback allowed.'],curobo_matching_source_files=matched,inference=dict(denoise_steps=10,denoise_mode='sync',compile=True,dit_cache=True,cosine_threshold=.99,max_skips=3,inference_horizon='native_full'),criterion=dict(clean_target_pct=93.74,randomized_target_pct=93.46,mean_target_pct=93.60,absolute_tolerance_pp=2,complete_suite_required=True),original_attempts_unchanged=True)
write(ROOT/'protocol.json',p)
argv=['/usr/bin/python3',str(ROOT/'run_native.py')];subprocess.run(argv+['--check-admission'],check=True)
job=dict(id='ex8a-native-official-pilot',status='ready',scientific_identity=f'{ROOT.name}/native-official-readiness/seed0',max_seconds=p['max_seconds'],authorization_evidence=str(ROOT/'protocol.json'),authorization_sha256=sha(ROOT/'protocol.json'),pause_paths=[str(ROOT/'paused.json'),'/home/zifanz4/.local/state/openwam-selfcheck/paused.json'],host_commands={'gpu01':dict(argv=argv,admission_argv=argv+['--check-admission'],cwd=str(ROOT),input_sha256={str(ROOT/'protocol.json'):sha(ROOT/'protocol.json')},outputs_absent=[str(ROOT/'pilot')])})
write(ROOT/'registry.json',dict(name=ROOT.name,question='Native released checkpoint reproduction before representation comparisons',stage='registered_L1_L3',protocol_sha256=sha(ROOT/'protocol.json'),job=job))
dest=R/'nightly-gpu/jobs'/f'{job["id"]}.json';assert not dest.exists();tmp=dest.with_suffix('.pending');write(tmp,job);tmp.replace(dest)
print(json.dumps(dict(registered=job['id'],textures=len(textures),curobo_sources=matched,protocol_sha256=sha(ROOT/'protocol.json'))))
