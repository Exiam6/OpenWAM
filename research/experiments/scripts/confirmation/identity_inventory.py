import sys,json,os,re,hashlib
from pathlib import Path
R=Path(sys.argv[1]);hashes=set();checked=[];errors=[]
def visit(v,key=''):
 if isinstance(v,dict):
  for k,x in v.items():visit(x,k)
 elif isinstance(v,list):
  for x in v:visit(x,key)
 elif isinstance(v,str) and re.fullmatch('[a-f0-9]{64}',v) and any(x in key.lower() for x in ['hash','sha256']):hashes.add(v)
for base in [R/'results',R/'data']:
 for root,dirs,files in os.walk(base,followlinks=False):
  dirs[:]=[d for d in dirs if d!='confirmation-20260923']
  for name in files:
   p=Path(root)/name
   if p.suffix=='.json' and p.stat().st_size<=20*1024*1024:
    try:
     raw=p.read_bytes();visit(json.loads(raw));checked.append((str(p),hashlib.sha256(raw).hexdigest()))
    except Exception as e:errors.append((str(p),repr(e)))
assert not errors,errors
print(json.dumps({'root':str(R),'hashes':sorted(hashes),'files':len(checked),'inventory_sha256':hashlib.sha256(json.dumps(sorted(checked)).encode()).hexdigest(),'scope':'recorded prior image/episode/file hashes; seeds independently excluded'}))
