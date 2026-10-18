from pathlib import Path
import subprocess,tarfile,io,json,hashlib,datetime
V=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018');base=Path('/home/zifanz4/openwam-rae-research');target=V/'corrected-engine-v1';target.mkdir(exist_ok=False);commit='e983676b069150d0346fcd9e90fd8ebe8a840f13'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip()==commit
assert not subprocess.check_output(['git','status','--porcelain'],cwd=base,text=True).strip()
blob=subprocess.check_output(['git','archive','--format=tar',commit],cwd=base)
with tarfile.open(fileobj=io.BytesIO(blob)) as tar:tar.extractall(target,filter='data')
p=json.loads((V/'gpu-registration.json').read_text());prefix=str(base)+'/'
registered={path:digest for path,digest in p['files'].items() if path.startswith(prefix)}
for path,digest in registered.items():
 assert sha(path)==digest,path
 assert sha(target/Path(path).relative_to(base))==digest,path
changed='openwam/model/video_backbone/wan/models/vae.py';candidate=Path('/home/zifanz4/openwam-vae-control')/changed
old=(base/changed).read_text();new=candidate.read_text();assert old!=new
# Ensure the tested candidate module is exactly the registered encoder-only replacement.
assert sha(candidate)==json.loads((V/'fixed-policy-diagnostic-0825/plan.json').read_text())['frozen_files'][str(candidate)]
(target/changed).write_bytes(candidate.read_bytes())
rows=[]
for path,digest in registered.items():
 rel=str(Path(path).relative_to(base));actual=sha(target/rel)
 assert (actual!=digest)==(rel==changed)
 rows.append({'relative_path':rel,'original_sha256':digest,'candidate_sha256':actual,'changed':actual!=digest})
record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':commit,'snapshot':str(target),'production_delta':[changed],'registered_source_files_verified':len(rows),'candidate_file_source':str(candidate),'candidate_fix_commit':'00319e8c','original_runtime_unchanged':True,'full_training_released':False,'rows':rows,'scope':'CPU-only prepared runtime snapshot, derived directly from the registered production commit. The broader openwam-vae-control worktree is not used as the runtime because it differs in other production modules. Only the previously isolated and tested VAE method change is included.'}
(V/'corrected-engine-v1-manifest.json').write_text(json.dumps(record,indent=2)+'\n');print({k:v for k,v in record.items() if k!='rows'})
