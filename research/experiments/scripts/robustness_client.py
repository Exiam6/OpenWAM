"""Run unmodified RoboTwin evaluation with an observation-only intervention."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
from robustness_common import change_observation, digest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'OpenWAM/benchmarks/robotwin'))
from eval_policy_wrapper import bootstrap_robotwin_module, _install_test_num_override

def main():
    out = Path(os.environ['ROBUST_RUN_DIR']); out.mkdir(parents=True, exist_ok=True)
    condition = os.environ['ROBUST_CONDITION']
    module = bootstrap_robotwin_module()
    _install_test_num_override(module)
    import openwam2robotwin_interface as interface
    old_eval, old_class = interface.eval, module.class_decorator
    records = []
    current = None
    last_seed = None

    # RoboTwin seeds numpy/torch, but instruction generation uses Python random.
    # Restore its state so the diagnostic does not alter the environment RNG.
    old_generate = module.generate_episode_descriptions
    def descriptions(*args, **kwargs):
        state = random.getstate()
        try:
            random.seed(last_seed)
            return old_generate(*args, **kwargs)
        finally:
            random.setstate(state)
    module.generate_episode_descriptions = descriptions

    def eval_step(env, model, observation):
        nonlocal current
        step = int(env.take_action_cnt)
        if current is None:
            current = {'seed': last_seed, 'instruction': env.get_instruction(),
                       'initial_head_sha256': digest(observation['observation']['head_camera']['rgb']),
                       'initial_state_sha256': digest(interface._extract_proprio(model, observation)),
                       'condition': condition, 'calls': 0}
        altered = change_observation(observation, condition, last_seed, step)
        if current['calls'] == 0:
            from PIL import Image
            for label, ob in [('clean', observation), ('policy', altered)]:
                Image.fromarray(ob['observation']['head_camera']['rgb']).save(out/f'seed-{last_seed}-{label}.png')
            current['policy_head_sha256'] = digest(altered['observation']['head_camera']['rgb'])
        old_eval(env, model, altered)
        current['calls'] += 1
        current['actions'] = int(env.take_action_cnt)
        current['success'] = bool(env.eval_success)
        (out/'active-episode.json').write_text(json.dumps(current, indent=2))

    def create(task):
        env = old_class(task)
        setup, close = env.setup_demo, env.close_env
        def setup_demo(*args, **kwargs):
            nonlocal last_seed
            last_seed = int(kwargs['seed'])
            return setup(*args, **kwargs)
        def close_env(*args, **kwargs):
            nonlocal current
            if current is not None:
                records.append(current)
                (out/'episodes.json').write_text(json.dumps(records, indent=2))
                print('ROBUST_EPISODE', json.dumps(current), flush=True)
                current = None
            return close(*args, **kwargs)
        env.setup_demo, env.close_env = setup_demo, close_env
        return env

    interface.eval, module.class_decorator = eval_step, create
    args = module.parse_args_and_config()
    module.main(args)
    assert len(records) == int(os.environ['ROBOTWIN_TEST_NUM'])
    assert all(r['calls'] == r['actions'] and r['actions'] > 0 for r in records)
    summary = {'condition': condition, 'successes': sum(r['success'] for r in records),
               'episodes': len(records), 'records': records,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('ROBUST_COMPLETE', json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
