"""Fresh expert reference or fixed-manifest replay; observation-only stress."""
import ast, hashlib, inspect, json, os
from pathlib import Path
import robustness_client as client
from robustness_common import noise_image
def install_replay(module, scenes):
    assert len(scenes)>=1 and len({x['seed'] for x in scenes})==len(scenes)
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
    # Keep the original reference NumPy choice length and its RNG consumption, while
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
        assert test_num==len(scenes)
        row=by_seed[active['seed']];prompt=row['instruction']
        count=row.get('instruction_choice_length',len(scenes))
        return [{'seen':[prompt]*count,'unseen':[prompt]*count}]
    module.class_decorator=create
    module.generate_episode_descriptions=descriptions
    return module


def main():
    manifest=os.environ.get('NORM_SCENES')
    by_seed=None
    if manifest:
        scenes=json.loads(Path(manifest).read_text())['scenes'];by_seed={r['seed']:r for r in scenes}
        assert len(scenes)==int(os.environ['ROBOTWIN_TEST_NUM'])
        original_bootstrap=client.bootstrap_robotwin_module
        client.bootstrap_robotwin_module=lambda:install_replay(original_bootstrap(),scenes)
    else:
        original_bootstrap=client.bootstrap_robotwin_module
        def reference_bootstrap():
            module=original_bootstrap();generate=module.generate_episode_descriptions
            def descriptions(*args,**kwargs):
                result=generate(*args,**kwargs)
                assert len(result[0]['unseen'])==10, 'Reference language choice count differs from frozen replay'
                return result
            module.generate_episode_descriptions=descriptions
            return module
        client.bootstrap_robotwin_module=reference_bootstrap
    sigma=float(os.environ['NORM_SIGMA']);assert sigma in (0.,.20)
    def change(observation,condition,seed,step):
        if by_seed is not None and step==0:
            import openwam2robotwin_interface as interface
            row=by_seed[seed]
            assert client.digest(observation['observation']['head_camera']['rgb'])==row['initial_head_sha256'],f'Initial image mismatch {seed}'
            assert client.digest(interface._extract_eef_proprio(observation))==row['initial_state_sha256'],f'Initial state mismatch {seed}'
        if condition=='clean':
            assert sigma==0.
            return observation
        assert condition=='noise' and sigma==.20
        cameras=observation['observation']
        rgb=noise_image(cameras['head_camera']['rgb'],seed,step,sigma=sigma)
        return {**observation,'observation':{**cameras,'head_camera':{**cameras['head_camera'],'rgb':rgb}}}
    client.change_observation=change
    out=Path(os.environ['ROBUST_RUN_DIR']);out.mkdir(parents=True,exist_ok=True)
    info={'arm':os.environ['NORM_ARM'],'noise_sigma':sigma,'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest() if manifest else None,'role':os.environ['NORM_ROLE'],'expert_filter':'reference once' if not manifest else 'no repeated filter, frozen manifest','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'provenance.json').write_text(json.dumps(info,indent=2));client.main()
if __name__=='__main__':main()
