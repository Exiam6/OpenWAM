#!/usr/bin/env python3
"""Exercise the actual upstream language generator and the proposed scoped fix."""
import argparse,ast,difflib,hashlib,importlib.util,json,random,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--robotwin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 source=a.robotwin/'script/eval_policy.py';old=source.read_text()
 needle='        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)'
 replacement='''        # Pair language with the scene seed without changing Python's outer RNG.
        language_rng_state = random.getstate()
        try:
            random.seed(now_seed)
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        finally:
            random.setstate(language_rng_state)'''
 assert old.count(needle)==1
 new=old.replace('import sys\n','import sys\nimport random\n',1).replace(needle,replacement)
 patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/script/eval_policy.py',tofile='b/script/eval_policy.py'))
 target=a.output/'robotwin-instruction-seed.patch';target.write_text(patch)
 subprocess.run(['git','-C',str(a.robotwin),'apply','--check',str(target.resolve())],check=True)
 path=a.robotwin/'description/utils/generate_episode_instructions.py'
 spec=importlib.util.spec_from_file_location('upstream_language_generator',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 tree=ast.parse(new);ev=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval_policy');loop=next(n for n in ev.body if isinstance(n,ast.While))
 begin=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='language_rng_state' for t in n.targets))
 fragment=loop.body[begin:begin+2]
 ns={'random':random,'generate_episode_descriptions':module.generate_episode_descriptions,'args':{'task_name':'pick_dual_bottles'},'episode_info_list':[{'{A}':'001_bottle/base13','{B}':'001_bottle/base16'}],'test_num':5,'now_seed':200001}
 original=[];fixed=[]
 fingerprint=lambda r:hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
 for i in range(5):
  random.seed(800+i);original.append(fingerprint(module.generate_episode_descriptions(ns['args']['task_name'],ns['episode_info_list'],5)))
  random.seed(800+i);prior=random.getstate()
  exec(compile(ast.fix_missing_locations(ast.Module(body=fragment,type_ignores=[])),str(target),'exec'),ns)
  assert random.getstate()==prior
  fixed.append(fingerprint(ns['results']))
 assert len(set(original))>1 and len(set(fixed))==1
 record={'upstream_eval_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'upstream_generator_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'unseeded_unique_lists':len(set(original)),'fixed_unique_lists':len(set(fixed)),'original_hashes':original,'fixed_hashes':fixed,'outer_rng_preserved':True,'tested_exact_patched_ast_block':True,'patch_applies_cleanly':True}
 (a.output/'robotwin-instruction-seed-reproduction.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
