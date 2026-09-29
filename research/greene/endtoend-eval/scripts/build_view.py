"""Greene path view for the endtoend checkpoints (no frozen file is modified).

Recreates the original /data02/zifanz4/openwam-experiments layout as a symlink
union over the three byte-verified asset roots, then gives each of the nine
checkpoints a view directory whose config.yaml has the single prefix
/data02/zifanz4/openwam-experiments -> ROOT substituted. Every other file in the
view is a symlink to the verified original. Idempotent; refuses to change an
existing different link.
"""
import hashlib,json,datetime
from pathlib import Path
RT=Path('/scratch/zz4330/openwam-runtime');ROOT=RT/'endtoend-root';VIEW=RT/'endtoend-eval/ckpt'
OLD='/data02/zifanz4/openwam-experiments'
AE=RT/'assets-endtoend';AW=RT/'assets-wan48'
REC=Path('/scratch/zz4330/OpenWAM/research/greene/records/endtoend-view.json')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<24),b''):h.update(b)
 return h.hexdigest()
def link(p,target):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.is_symlink():assert p.resolve()==target.resolve(),(p,target);return
 assert not p.exists(),p;assert target.exists(),target;p.symlink_to(target)
def frozen():
 want={}
 for f in ['wan48','endtoend']:
  for line in Path(f'/scratch/zz4330/OpenWAM/research/greene/delta/{f}/SHA256SUMS').read_text().split('\n'):
   if line.strip():h,rel=line.split(None,1);want[rel.strip().lstrip('*').removeprefix('./')]=h
 return want
def main():
 link(ROOT/'assets',AE/'assets');link(ROOT/'results',AW/'results');link(ROOT/'benchmarks/RoboTwin',RT/'benchmarks/RoboTwin')
 E=ROOT/'endtoend-20260923';link(E/'normalization_stats.npy',AW/'endtoend-20260923/normalization_stats.npy')
 for d in (AE/'endtoend-20260923').iterdir():
  if d.name!='training':link(E/d.name,d)
 want=frozen();ckpts={}
 for route in ['wan','svae','pca']:
  for seed in [42,43,44]:
   run=f'{route}-seed{seed}';src=(AW if route=='wan' else AE)/'endtoend-20260923/training'/run
   link(E/'training'/run,src)
   (out,)=list((src/'output').iterdir());ck=out/'checkpoint_step_12000.safetensors'
   rel=str(ck.relative_to(src.parents[2]));h=sha(ck)
   assert want.get(rel)==h,(run,rel,h)
   v=VIEW/run;v.mkdir(parents=True,exist_ok=True)
   for f in out.iterdir():
    if f.name!='config.yaml':link(v/f.name,f)
   text=(out/'config.yaml').read_text();new=text.replace(OLD,str(ROOT));assert '/data02' not in new and '/home/zifanz4' not in new
   cfg=v/'config.yaml'
   if cfg.exists():assert cfg.read_text()==new,cfg
   else:cfg.write_text(new)
   for line in new.split('\n'):
    if str(ROOT) in line:
     p=Path(line.split(':',1)[1].strip())
     if not any(k in line for k in ['output_path','output_dir']):assert p.exists(),(run,line)
   ckpts[run]={'view':str(v),'checkpoint':str(ck),'checkpoint_sha256':h,'frozen_sha256_match':True,'original_config_sha256':sha(out/'config.yaml'),'view_config_sha256':sha(cfg)}
   print(run,h[:12],'ok',flush=True)
 REC.write_text(json.dumps({'built_at':datetime.datetime.now().astimezone().isoformat(),'root':str(ROOT),'prefix_substitution':[OLD,str(ROOT)],'checkpoints':ckpts},indent=2)+'\n')
if __name__=='__main__':main()
