"""Read-only seed inventory excluding this prospective study's plan files."""
import os,json,re,datetime,sys,hashlib
from pathlib import Path
R=Path(sys.argv[1]);dest=Path(sys.argv[2]);proposed=set(range(1100000,1100100))|set(range(1101000,1101100))|set(range(1102000,1102100));seen=set();matches=[];checked=[];errors=[]
def visit(value,key=''):
 if isinstance(value,dict):
  for k,v in value.items():visit(v,k)
 elif isinstance(value,list):
  for v in value:visit(v,key)
 elif isinstance(value,int) and not isinstance(value,bool) and 'seed' in key.lower():
  seen.add(value)
  if value in proposed:matches.append({'file':str(p),'key':key,'seed':value})
for base in [R/'results',R/'data']:
 for root,dirs,files in os.walk(base,followlinks=False):
  dirs[:]=list(dirs)
  for d in dirs:
   m=re.fullmatch(r'seed[-_]?(\d+)',d)
   if m:
    n=int(m.group(1));seen.add(n)
    if n in proposed:matches.append({'directory':str(Path(root)/d),'seed':n})
  for f in files:
   p=Path(root)/f
   if f.endswith('.json'):
    if p.stat().st_size>20*1024*1024:continue
    try:raw=p.read_bytes();visit(json.loads(raw));checked.append((str(p.relative_to(R)),hashlib.sha256(raw).hexdigest()))
    except (OSError,ValueError) as e:errors.append({'file':str(p),'error':repr(e)})
   elif f.lower() in ['seed.txt','seeds.txt']:
    raw=p.read_bytes();checked.append((str(p.relative_to(R)),hashlib.sha256(raw).hexdigest()))
    for token in re.findall(r'\d+',raw.decode()):
     n=int(token);seen.add(n)
     if n in proposed:matches.append({'file':str(p),'seed':n})
report={'time':datetime.datetime.now().astimezone().isoformat(),'passed':not matches and not errors,'scanned_root':str(R),'json_seed_fields_and_seed_directories':True,'known_seed_count':len(seen),'known_seed_min_max':[min(seen),max(seen)] if seen else [],'proposed_ranges':[[1100000,1100099],[1101000,1101099],[1102000,1102099]],'files_checked':len(checked),'inventory_sha256':hashlib.sha256(json.dumps(sorted(checked)).encode()).hexdigest(),'matches':matches,'errors':errors,'scope':'all prior results/data, including completedconfirmation; new endtoend outside these roots; content-identity check separately required before scoring'}
dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));assert report['passed']
