"""Native collection and isolated renderer checks for the new confirmation."""
import argparse, datetime, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

R = Path(os.environ.get('WAM_ROOT', '/data02/zifanz4/openwam-experiments'))
O = R / 'results/confirmation-20260923'
sys.path.insert(0, str(R / 'scripts/goalaux'))

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(p)

def now(): return datetime.datetime.now().astimezone()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def guards():
    for p in [O, R, Path('/home/zifanz4/openwam-experiments/studies/confirmation-20260923'), Path('/home/zifanz4/.local/state/openwam-selfcheck')]:
        assert not (p / 'paused.json').exists(), str(p)
    assert now() < datetime.datetime.fromisoformat(json.loads((O / 'window.json').read_text())['deadline'])

def preflight(slot):
    guards(); dest = O / 'preflight' / slot; dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'result.json').exists()
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_name(0) == 'NVIDIA L40S'
    torch.zeros(1, device='cuda')
    sys.path.insert(0, str(R / 'OpenWAM/benchmarks/robotwin'))
    from eval_policy_wrapper import bootstrap_robotwin_module
    bootstrap_robotwin_module()
    from curobo.wrap.reacher.motion_gen import MotionGen
    import curobo, sapien, numpy as np, importlib.metadata
    assert str(curobo.__file__).startswith(str(R / 'benchmarks/RoboTwin'))
    engine = sapien.Engine(); renderer = sapien.SapienRenderer(); engine.set_renderer(renderer)
    sapien.render.set_camera_shader_dir('rt')
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser('oidn')
    scene = engine.create_scene(); device = scene.render_system.device
    from collect_confirmation_v2 import pci
    assert pci(device.pci_string) == pci(os.environ['EXPECTED_RENDER_PCI'])
    scene.set_ambient_light([.5, .5, .5]); scene.add_ground(0)
    camera = scene.add_camera('preflight', 64, 64, 1, .1, 10)
    camera.set_pose(sapien.Pose([0, 0, 1])); scene.step(); scene.update_render(); camera.take_picture()
    rgba = camera.get_picture('Color'); assert rgba.shape == (64, 64, 4) and np.isfinite(rgba).all()
    write(dest / 'result.json', {'passed': True, 'completed_at': now().isoformat(), 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'renderer_pci': device.pci_string, 'renderer_name': device.name, 'versions': {k: importlib.metadata.version(k) for k in ['torch', 'sapien', 'h5py', 'numpy']}, 'no_confirmation_seed_used': True, 'planner_per_scene_gate_still_required': True})

def one(task, seed):
    import collect_confirmation_v2 as ref
    ref.ROOT = R; ref.RUN = O / 'fresh'
    ref.worker(task, seed)

def collect(task):
    from null_grasp_classification_v2 import annotate
    guards(); plan = json.loads((O / 'protocol.json').read_text())
    assert json.loads((O / 'preflight' / task / 'result.json').read_text())['passed']
    assert json.loads((O / 'seed-exclusion.json').read_text())['passed']
    for path, expected in json.loads((O / 'freeze.json').read_text())['files'].items():
        assert sha(path) == expected, path
    dest = O / 'fresh' / task; dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'started.json').exists(), 'No automatic collection restart'
    write(dest / 'started.json', {'time': now().isoformat(), 'gpu': os.environ['CUDA_VISIBLE_DEVICES']})
    deadline = datetime.datetime.fromisoformat(plan['collection_deadline'])
    attempts = []; accepted = []; start = time.monotonic()
    for offset in range(plan['max_candidates_per_task']):
        guards(); remaining = (deadline - now()).total_seconds(); assert remaining > 0, 'Collection cutoff reached'
        seed = plan['new_seed_starts'][task] + offset
        run = dest / f'seed-{seed}'; run.mkdir(exist_ok=False)
        with (run / 'worker.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', __file__, 'one', '--task', task, '--seed', str(seed)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(run / 'owned-worker.json', {'pid': proc.pid, 'process_group': proc.pid, 'seed': seed, 'task': task})
            try: code = proc.wait(timeout=min(600, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
                code = 124
        f = run / 'result.json'
        raw = json.loads(f.read_text()) if f.exists() else {'task': task, 'seed': seed, 'accepted': False, 'outcome': 'infrastructure_failure', 'error': 'No result from worker'}
        raw['exit_code'] = code; result = annotate(raw); attempts.append(result)
        if result.get('classification_amendment'): write(run / 'classification-amendment.json', result)
        if result.get('accepted'):
            assert result['frames'] >= 44, 'Complete scene too short; invalid cohort, no replacement'
            accepted.append(result)
        write(dest / 'progress.json', {'attempts': attempts, 'accepted': accepted, 'accepted_count': len(accepted), 'elapsed_seconds': time.monotonic() - start, 'collection_deadline': deadline.isoformat()})
        print(task, seed, result['outcome'], len(accepted), flush=True)
        assert code == 0 and result['outcome'] != 'infrastructure_failure', 'Unclassified failure retained; no retry'
        if len(accepted) == plan['accepted_per_task']: break
    write(dest / 'manifest.json', {'task': task, 'accepted': accepted, 'attempts': attempts, 'complete': len(accepted) == plan['accepted_per_task'], 'completed_at': now().isoformat(), 'collection_deadline': deadline.isoformat()})
    assert len(accepted) == plan['accepted_per_task'], 'Incomplete; retain without scoring'

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['preflight', 'collect', 'one']); p.add_argument('--task', required=True); p.add_argument('--seed', type=int)
    a = p.parse_args()
    if a.mode == 'preflight': preflight(a.task)
    elif a.mode == 'collect': collect(a.task)
    else: one(a.task, a.seed)
