"""CPU-only audit of existing expanded-study data and deployment contracts.

Run with --output PATH. Reads GPU02 artifacts; never loads a policy or simulator.
Only the small proprio/timestamp members of trajectory NPZs are decompressed.
"""

import argparse
import json
import subprocess
from pathlib import Path

REMOTE = r'''
import ast, collections, datetime, hashlib, importlib.util, json, logging, os, pathlib, sys
from typing import Optional
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
root = pathlib.Path('/home/zifanz4/openwam-runtime/rae-policy-expanded-20261006')
source = pathlib.Path('/home/zifanz4/openwam-rae-research')
sys.path.insert(0, str(source))
import numpy as np
from omegaconf import OmegaConf, DictConfig
from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.utils.normalization import pin_rot6d_identity, ROT6D_DIMS_EEF20

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def summary(x):
    a = np.asarray(x)
    return {'count': int(a.size), 'min': float(a.min()), 'median': float(np.median(a)), 'max': float(a.max())}

protocol = json.loads((root/'protocol.json').read_text())
collection = json.loads((root/'collection-result.json').read_text())
valid = [r for r in collection['records'] if r['status'] == 'complete']
training_states, episode_rows = [], []
timing = {t: {'intervals': [], 'horizons': [], 'duplicate_pairs': 0, 'windows_with_duplicate': 0,
              'covered_actions': 0, 'eligible_actions': 0} for t in protocol['tasks']}
for r in valid:
    path = root/'data'/r['id']/'full-trajectory.npz'
    with np.load(path, allow_pickle=False) as data:
        q, ticks, dt = data['proprio'], data['physics_ticks'], float(data['timestep'])
    assert len(q) == len(ticks) == r['frames'] and q.shape[1] == 20
    assert np.isfinite(q).all() and (np.diff(ticks) >= 0).all()
    if r['split'] != 'train':
        continue
    training_states.append(q)
    starts = np.unique(np.linspace(0, len(q)-33, protocol['windows_per_episode']['train']).astype(int))
    diffs = np.diff(ticks)
    spans = (ticks[starts+32]-ticks[starts])*dt
    t = timing[r['task']]
    t['intervals'].extend((diffs*dt).tolist())
    t['horizons'].extend(spans.tolist())
    t['duplicate_pairs'] += int((diffs == 0).sum())
    t['windows_with_duplicate'] += sum(bool((diffs[s:s+32] == 0).any()) for s in starts)
    covered = {i for s in starts for i in range(s+1, s+33)}
    t['covered_actions'] += len(covered)
    t['eligible_actions'] += len(q)-1
    episode_rows.append({'id': r['id'], 'frames': len(q), 'state_sha256': hashlib.sha256(q.tobytes()).hexdigest(),
                         'ticks_sha256': hashlib.sha256(ticks.tobytes()).hexdigest(), 'windows': len(starts)})
q = np.concatenate(training_states)
stats = dict(mean=q.mean(0), std=q.std(0), min=q.min(0), max=q.max(0),
             q01=np.quantile(q, .01, axis=0), q99=np.quantile(q, .99, axis=0))
pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
saved = np.load(root/'normalization_stats.npy', allow_pickle=True).item()
assert saved['num_timesteps'] == len(q)
assert all(np.array_equal(stats[k], saved['eef'][k]) for k in stats)
train_norm = Normalizer('min_max', stats)
normalized = train_norm.normalize(q)
roundtrip = train_norm.unnormalize(normalized)
error = float(np.abs(roundtrip-q).max())
assert error < 1e-5, error

# Execute the actual deployment normalizer builder without importing model-loading
# machinery. AST extraction preserves its body; the tested checkpoint uses no
# legacy command-aware branch (explicitly asserted below).
loader_path = source/'openwam/deploy/model_loader.py'
tree = ast.parse(loader_path.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_build_inner_normalizer')
namespace = {'OmegaConf': OmegaConf, 'DictConfig': DictConfig, 'os': os, 'logger': logging.getLogger('audit')}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(loader_path), 'exec'), namespace)
runs = []
for arm in protocol['variants']:
    for seed in protocol['seeds']:
        run = root/'runs'/f'{arm}-seed{seed}'
        cfg = OmegaConf.load(run/'config.yaml')
        dl = cfg.dataloader
        assert dl.action_mode == 'eef' and not dl.unify_action and dl.normalize_mode == 'min-max'
        assert not OmegaConf.select(cfg, 'dataloader.binary_action_dims')
        assert OmegaConf.select(cfg, 'dataloader.base_proprio', default='velocity') == 'velocity'
        assert dl.num_frames == 33 and dl.video_stride == 4 and dl.camera_layout == protocol['camera_layout']
        assert dl.height == 384 and dl.width == 320
        assert sha(run/'normalization_stats.npy') == sha(root/'normalization_stats.npy')
        deployed = namespace['_build_inner_normalizer'](cfg, str(run))
        assert np.array_equal(deployed.normalize(q), normalized)
        assert np.array_equal(deployed.unnormalize(normalized), roundtrip)
        runs.append({'run': run.name, 'normalization_sha256': sha(run/'normalization_stats.npy'),
                     'config_sha256': sha(run/'config.yaml'), 'normalize_exact': True, 'unnormalize_exact': True})

# Exercise the real executor with synthetic chunks, without model inference.
executor_path = source/'openwam/deploy/executors/sync_executor.py'
executor_ast = next(n for n in ast.parse(executor_path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'SyncInferenceExecutor')
executor_ns = {'np': np, 'deque': collections.deque, 'Optional': Optional, 'BaseInferenceEngine': object}
exec(compile(ast.Module(body=[executor_ast], type_ignores=[]), str(executor_path), 'exec'), executor_ns)
class FakeEngine:
    def __init__(self): self.calls = 0
    def generate(self, conditions):
        start = self.calls*100
        self.calls += 1
        return {'actions': np.repeat((start+np.arange(32))[:, None], 20, axis=1)}
engine = FakeEngine()
executor = executor_ns['SyncInferenceExecutor'](engine, inference_horizon=8)
observed = [int(executor.predict_action({})[0]) for _ in range(17)]
assert observed == [*range(8), *range(100,108), 200] and engine.calls == 3
executor.reset()
assert int(executor.predict_action({})[0]) == 300 and engine.calls == 4
bad = OmegaConf.create(OmegaConf.to_container(cfg))
bad.dataloader.normalize_mode = 'unsupported-mode'
try: namespace['_build_inner_normalizer'](bad, str(run))
except ValueError: pass
else: raise AssertionError('Invalid normalization mode accepted')

client = pathlib.Path('/home/zifanz4/openwam-runtime/assets-source/temporal-20260926/OpenWAM/benchmarks/robotwin')
prompt_path = client/'prompt_template.py'
spec = importlib.util.spec_from_file_location('audit_prompt', prompt_path)
prompt = importlib.util.module_from_spec(spec); spec.loader.exec_module(prompt)
evaluation = json.loads((root/'eval-protocol.json').read_text())
assert all(prompt.format_prompt_for_inference(evaluation['instructions'][t]) == protocol['prompts'][t] for t in protocol['tasks'])

helper = pathlib.Path('/home/zifanz4/OpenWAM/openwam/dataloader/transforms/multiview.py')
deploy_helper = source/'openwam/dataloader/transforms/multiview.py'
def function_ast(path, name):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
layout_function_equal = {name: function_ast(helper, name) == function_ast(deploy_helper, name)
                         for name in ['assemble_multiview_layout', 'crop_and_resize']}
assert all(layout_function_equal.values())
sim = root.parent/'assets-source/benchmarks/RoboTwin/envs/_base_task.py'
paths = [root/'prepare.py', root/'train.py', root/'common.py', root/'collect.py', root/'serve_policy.py',
         root/'evaluate_policy.py', root/'protocol.json', root/'eval-protocol.json', helper, deploy_helper,
         loader_path, source/'openwam/dataloader/transforms/normalize.py', source/'openwam/dataloader/utils/normalization.py',
         source/'openwam/dataloader/robotwin.py', source/'openwam/deploy/executors/sync_executor.py',
         source/'openwam/deploy/server.py', source/'openwam/deploy/engine.py', source/'configs/deploy.yaml',
         prompt_path, client/'openwam2robotwin_interface.py', sim]
out = {'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'scope': 'CPU read-only checks on existing data; CUDA hidden; no model or simulator instantiated',
       'checks': {'training_episodes': len(training_states), 'training_states': len(q),
                  'train_only_stats_exact': True, 'all_nine_run_normalizers_exact': True,
                  'roundtrip_max_abs': error, 'prompts_exact': True, 'layout_function_ast_equal': layout_function_equal},
       'timing': {}, 'runs': runs, 'source_hashes_at_audit': {str(p): sha(p) for p in paths},
       'episodes': episode_rows,
       'limits': ['NPZ proprio/timestamp members read; compressed RGB and 15 GB feature cache not loaded or rehashed.',
                  'Current source hashes record audit provenance; not all dependencies were in the original 21-file freeze.',
                  'Nonuniform expert recording cadence also exists in native RoboTwin; not proof of a new regression.',
                  'No causal explanation of closed-loop failures is established by these checks.']}
out['checks']['sync_executor_horizon8_tail_discard_and_reset'] = True
out['checks']['invalid_normalization_mode_rejected'] = True
for task, t in timing.items():
    out['timing'][task] = {'record_interval_seconds': summary(t['intervals']),
                          'action_32_horizon_seconds': summary(t['horizons']),
                          'duplicate_intervals': t['duplicate_pairs'],
                          'windows_with_duplicate': t['windows_with_duplicate'],
                          'covered_action_indices': t['covered_actions'], 'eligible_action_indices': t['eligible_actions'],
                          'coverage_fraction': t['covered_actions']/t['eligible_actions']}
print(json.dumps(out))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'gpu02',
         '/home/zifanz4/openwam-runtime/policy-env/bin/python', '-'],
        input=REMOTE, text=True, capture_output=True, check=True, timeout=180,
    )
    report = json.loads(result.stdout)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('checks', 'timing')}, indent=2))


if __name__ == '__main__':
    main()
