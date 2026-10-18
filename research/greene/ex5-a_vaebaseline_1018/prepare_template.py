"""Prepare configuration only; do not construct a model or initialize CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from omegaconf import OmegaConf

R=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
B=R.parent/'rae-factorial-20261010/templates/raw768_native'
D=R/'templates/vae48_native'
assert not D.exists()
base=OmegaConf.to_container(OmegaConf.load(B/'config.yaml'),resolve=True)
config=copy.deepcopy(base)
config['model']['video_backbone']['encoder']={'name':'wan22_vae','model_path':str(R/'assets')}
for component in config['model']['video_backbone']['components']:
    if component['attr']=='dit':
        component['extra_kwargs']['in_dim']=48
        component['extra_kwargs']['out_dim']=48
config['training']['output_path']=str(R/'runs')
config['project']['output_dir']=str(R/'hydra')

def flat(value,prefix=''):
    if isinstance(value,dict):
        result={}
        for key,v in value.items():result.update(flat(v,prefix+'.'+key if prefix else key))
        return result
    if isinstance(value,list):
        result={}
        for i,v in enumerate(value):result.update(flat(v,f'{prefix}.{i}'))
        return result
    return {prefix:value}
a=flat(base);b=flat(config)
diff=[dict(key=k,before=a.get(k),after=b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
allowed=['model.video_backbone.encoder.','model.video_backbone.components.0.extra_kwargs.in_dim','model.video_backbone.components.0.extra_kwargs.out_dim','training.output_path','project.output_dir']
assert all(any(x['key'].startswith(p) for p in allowed) for x in diff)
assert config['model']['video_backbone']['shift_video']==5 and config['model']['action_backbone']['shift_action']==5
assert config['training']['max_steps']==20000 and config['model']['architecture']['action_dim']==20
D.mkdir(parents=True);OmegaConf.save(OmegaConf.create(config),D/'config.yaml')
shutil.copytree(B/'tokenizer',D/'tokenizer')
files={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file()}
result=dict(passed=True,template=str(D),baseline=str(B),changes=diff,files=files,model_constructed=False,gpu_launched=False,
    remaining='Need explicit VAE frozen-weight loader, input cache, shared trainable initialization validation and staged GPU budget before training.')
(R/'template-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}))
