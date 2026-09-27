"""Replay one reference-feasible scene/prompt manifest in every confirmation arm.

The published rollout loop is retained. Feasibility is established by the first
reference run; re-running that stochastic filter separately would break pairing.
"""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import robustness_client as client

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/robustness-20260921'


def install_replay(module, scenes):
    assert len(scenes)==5 and len({x['seed'] for x in scenes})==5
    original=module.eval_policy
    tree=ast.parse(inspect.getsource(original))
    fn=tree.body[0]
    assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expert_check' for t in n.targets))
    assert isinstance(assignment.value,ast.Constant) and assignment.value.value is True
    assignment.value=ast.Constant(False)
    loop=next(n for n in fn.body if isinstance(n,ast.While))
    assert ast.unparse(loop.test)=='succ_seed < test_num'
    prefix=ast.parse('''assert test_num == len(_fixed_scenes)
now_seed = _fixed_scenes[succ_seed]["seed"]
episode_info = {"info": {}}
''').body
    loop.body=prefix+loop.body
    module.__dict__['_fixed_scenes']=scenes
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename,'exec'),module.__dict__)
    # Keep the original NumPy choice length (5) and its RNG consumption, while
    # replaying the exact unseen prompt generated for this scene in the reference.
    active={'seed':None}
    original_class=module.class_decorator
    def create(task):
        env=original_class(task);setup=env.setup_demo
        def reset(*args,**kwargs):
            active['seed']=int(kwargs['seed'])
            return setup(*args,**kwargs)
        env.setup_demo=reset
        return env
    by_seed={r['seed']:r for r in scenes}
    def descriptions(task,episode_info_list,test_num):
        assert test_num==5
        prompt=by_seed[active['seed']]['instruction']
        return [{'seen':[prompt]*5,'unseen':[prompt]*5}]
    module.class_decorator=create
    module.generate_episode_descriptions=descriptions
    return module


def main():
    scenes=json.loads((OUT/'fixed-scenes.json').read_text())['scenes']
    original_bootstrap=client.bootstrap_robotwin_module
    def bootstrap():return install_replay(original_bootstrap(),scenes)
    client.bootstrap_robotwin_module=bootstrap
    original_change=client.change_observation
    by_seed={r['seed']:r for r in scenes}
    def change(observation,condition,seed,step):
        if step==0:
            import openwam2robotwin_interface as interface
            row=by_seed[seed]
            assert client.digest(observation['observation']['head_camera']['rgb'])==row['initial_head_sha256'],f'Initial scene image mismatch: {seed}'
            assert client.digest(interface._extract_eef_proprio(observation))==row['initial_state_sha256'],f'Initial state mismatch: {seed}'
        return original_change(observation,condition,seed,step)
    client.change_observation=change
    out=Path(os.environ['ROBUST_RUN_DIR']);out.mkdir(parents=True,exist_ok=True)
    (out/'replay-provenance.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256((OUT/'fixed-scenes.json').read_bytes()).hexdigest(),'expert_filter':'accepted once in recorded reference; not re-filtered per condition','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    client.main()

if __name__=='__main__':main()
