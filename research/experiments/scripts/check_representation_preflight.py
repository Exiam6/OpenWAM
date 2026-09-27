"""Bounded CPU CLI and cached-sample alignment gates; never fit models."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import torch

ROOT = Path('/data02/zifanz4/openwam-experiments')
OUT = ROOT / 'results/representation-preflight-20260922'
TREE = Path('/home/zifanz4/openwam-svae-contribution')
PYTHON = ROOT / 'policy-env/bin/python'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    checkpoint = ROOT / 'results/night-20260921/adjust_bottle/svae-w1-s42/final.pt'
    feature = ROOT / 'results/night-20260921/adjust_bottle/features.pt'
    cache_path = ROOT / 'results/robustness-20260921/adapter-cache.pt'
    sources = [checkpoint, feature, cache_path, TREE / 'scripts/svae_train/evaluate_svae.py', ROOT / 'scripts/train_restoration.py']
    original = {str(p): digest(p) for p in sources}
    data = torch.load(feature, map_location='cpu', mmap=True, weights_only=False)
    indices = [i for i, row in enumerate(data['manifest']) if row['split'] == 'test'][:3]
    assert len(indices) == 3
    fixtures = OUT / 'native-shards'
    fixtures.mkdir(exist_ok=False)
    for part, ii in enumerate([indices[:2], indices[2:]]):
        torch.save({'features': data['features'][ii].clone(), 'raw_dim': data['features'].shape[1]}, fixtures / f'features_rank0_part{part}.pt')
    write('fixture-manifest.json', {'indices': indices, 'samples': [data['manifest'][i] for i in indices], 'scientific_evaluation': False})
    del data
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    runs = []
    for size in [2, 1]:
        command = [str(PYTHON), str(TREE / 'scripts/svae_train/evaluate_svae.py'), '--checkpoint', str(checkpoint), '--features-dir', str(fixtures), '--output', str(OUT / f'cli-batch{size}.json'), '--batch-size', str(size), '--device', 'cpu']
        with (OUT / f'cli-batch{size}.log').open('w') as f:
            run = subprocess.run(command, cwd=TREE, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=120)
        runs.append({'command': command, 'exit_code': run.returncode})
        write('cli-runs.json', runs)
        assert run.returncode == 0, f'CLI batch{size} failed'
    a, b = [json.loads((OUT / f'cli-batch{size}.json').read_text()) for size in [2, 1]]
    assert a['num_clips'] == b['num_clips'] == 3
    for key, value in a.items():
        if key.endswith('mse'):
            assert np.isfinite(value) and np.isclose(value, b[key], rtol=1e-6, atol=1e-9), key
    import sys
    sys.path.insert(0, str(TREE))
    from openwam.model.video_backbone.encoder.svae import load_svae
    model = load_svae(str(checkpoint))
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for key in ['input_mean', 'input_std']:
        assert torch.equal(getattr(model, key), ck['state_dict'][key])
    del model, ck
    cache = torch.load(cache_path, map_location='cpu', mmap=True, weights_only=False)
    manifest, refs = cache['manifest'], cache['references'].tolist()
    assert list(cache['clean'].shape) == [480, 48, 1, 24, 20]
    assert list(cache['noisy'].shape) == [900, 48, 1, 24, 20]
    assert len(manifest) == len(refs) == 900
    assert torch.isfinite(cache['clean']).all() and torch.isfinite(cache['noisy']).all()
    order = np.random.default_rng(42).permutation(50)
    splits = {int(e): 'train' if i < 35 else 'val' if i < 40 else 'test' for i, e in enumerate(order)}
    groups = defaultdict(list)
    for i, row in enumerate(manifest):
        assert row['split'] == splits[row['episode']] != 'test'
        groups[(row['episode'], row['frame'])].append(i)
    assert len(groups) == 480
    train = sorted({ep for ep, _ in groups if splits[ep] == 'train'})
    val = sorted({ep for ep, _ in groups if splits[ep] == 'val'})
    assert len(train) == 35 and len(val) == 5 and not set(train) & set(val)
    files = sorted((ROOT / 'data/pick_dual_bottles').rglob('*.hdf5'), key=lambda p: int(p.stem.replace('episode', '')))
    assert len(files) == 50 and all(p.stem == 'episode' + str(i) for i, p in enumerate(files))
    records, states, targets = [], [], []
    for episode in train + val:
        with h5py.File(files[episode], 'r') as f:
            n = len(f['endpose/left_endpose'])
            expected = np.unique(np.linspace(0, n - 33, 12, dtype=int)).tolist()
            actual = sorted(frame for ep, frame in groups if ep == episode)
            assert actual == expected
            for frame in actual:
                ii = groups[(episode, frame)]
                expected_rep = [0, 1] if splits[episode] == 'train' else [0]
                assert sorted(manifest[i]['replica'] for i in ii) == expected_rep
                reference = refs[ii[0]]
                assert all(refs[i] == reference for i in ii)
                assert 0 <= frame < frame + 4 < n
                current = np.concatenate([f['endpose/left_endpose'][frame], f['endpose/right_endpose'][frame], [f['endpose/left_gripper'][frame], f['endpose/right_gripper'][frame]]]).astype(np.float64)
                target = np.concatenate([f['endpose/left_endpose'][frame + 4, :3] - f['endpose/left_endpose'][frame, :3], f['endpose/right_endpose'][frame + 4, :3] - f['endpose/right_endpose'][frame, :3], [f['endpose/left_gripper'][frame + 4], f['endpose/right_gripper'][frame + 4]]]).astype(np.float64)
                assert current.shape == (16,) and target.shape == (8,)
                assert np.isfinite(current).all() and np.isfinite(target).all()
                images = {c: hashlib.sha256(bytes(f[f'observation/{c}/rgb'][frame])).hexdigest() for c in ['head_camera', 'left_camera', 'right_camera']}
                records.append({'episode': episode, 'frame': frame, 'target_frame': frame + 4, 'split': splits[episode], 'clean_index': reference, 'noisy_indices': ii, 'episode_file': str(files[episode].relative_to(ROOT / 'data')), 'current_state_sha256': array_hash(current), 'target_sha256': array_hash(target), 'encoded_image_sha256': images})
                states.append(current)
                targets.append(target)
    assert sorted(r['clean_index'] for r in records) == list(range(480))
    np.savez(OUT / 'aligned-state-targets.npz', state=np.stack(states), target=np.stack(targets))
    write('sample-manifest.json', records)
    unchanged = all(digest(Path(p)) == h for p, h in original.items())
    assert unchanged
    result = {'checked_at': datetime.now().astimezone().isoformat(), 'passed': True, 'cli': {'normal_package_bootstrap': True, 'calls': 2, 'clips_per_call': 3, 'batch_sizes': [2, 1], 'exit_codes': [0, 0], 'training_normalization_preserved': True, 'full_upstream_suite_run': False}, 'alignment': {'clean_samples': len(records), 'noisy_samples': len(manifest), 'train_episodes': train, 'validation_episodes': val, 'samples_by_split': dict(Counter(r['split'] for r in records)), 'test_episodes_opened': 0, 'raw_step_horizon': 4, 'state_dim': 16, 'target_dim': 8, 'timestamps': 'raw HDF5 row indices; not physical time'}, 'versions': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'diffusers', 'h5py', 'numpy']}, 'elapsed_seconds': time.monotonic() - start, 'inputs_unchanged': unchanged, 'source_hashes': original, 'limits': ['CLI fixture is software validation, not new held-out performance evidence.', 'Cache manifests/code align samples; original extraction did not store per-image hashes, so feature provenance cannot be retrospectively proven byte-for-byte without re-encoding.', 'Validation episodes were used for adapter selection; any later readout is development evidence.', 'Published representation may have trained on these demos; no matched pretrained comparison.', 'Targets are achieved state, not commands or contact; no rotation target.']}
    write('validation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}, indent=2))

if __name__ == '__main__':
    main()
