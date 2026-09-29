"""Frozen protocol CPU readout on existing representation caches."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from robustness_common import Adapter

ROOT = Path('/data02/zifanz4/openwam-experiments')
PRE = ROOT / 'results/representation-preflight-20260922'
OUT = ROOT / 'results/control-readout-20260922'
ALPHAS = [1e-4, .01, 1., 100.]
SEED = 20260922

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def fit(x, y, alpha):
    xm, xs = x.mean(0), x.std(0).clip(1e-6)
    ym, ys = y.mean(0), y.std(0).clip(1e-6)
    z, t = (x - xm) / xs, (y - ym) / ys
    w = np.linalg.solve(z.T @ z / len(z) + alpha * np.eye(z.shape[1]), z.T @ t / len(z))
    return dict(xm=xm, xs=xs, ym=ym, ys=ys, w=w)

def predict(m, x):
    return ((x - m['xm']) / m['xs']) @ m['w'] * m['ys'] + m['ym']

def select(x, y, episodes):
    ids = np.random.default_rng(SEED).permutation(np.unique(episodes))
    folds = np.array_split(ids, 5)
    scores = []
    for alpha in ALPHAS:
        errors = []
        for fold in folds:
            val = np.isin(episodes, fold)
            m = fit(x[~val], y[~val], alpha)
            errors.append(float(np.mean(((predict(m, x[val]) - y[val]) / m['ys']) ** 2)))
        scores.append({'alpha': alpha, 'fold_mse': errors, 'mean_mse': float(np.mean(errors))})
    alpha = min(scores, key=lambda s: s['mean_mse'])['alpha']
    return fit(x, y, alpha), {'alpha': alpha, 'scores': scores, 'validation_episodes_used': False, 'fold_episode_ids': [a.tolist() for a in folds]}

def train_only(x, y, episode, mask):
    return select(x[mask], y[mask], episode[mask])

def pooled(z):
    assert z.ndim == 5 and z.shape[2] == 1
    return F.adaptive_avg_pool2d(z[:, :, 0].float(), (2, 2)).flatten(1).numpy().astype(np.float64)

def metrics(pred, target, scale, episodes):
    error = pred - target
    normalized = (error / scale) ** 2
    values = {}
    for name, sl in [('all', slice(None)), ('translation', slice(0, 6)), ('gripper', slice(6, 8))]:
        values[name] = {'normalized_mse': float(normalized[:, sl].mean()), **({'rmse': float(np.sqrt(np.mean(error[:, sl] ** 2))) * (1000 if name == 'translation' else 1), 'rmse_unit': 'mm' if name == 'translation' else 'native gripper units'} if name != 'all' else {}),
                        'episode_normalized_mse': [float(normalized[episodes == ep, sl].mean()) for ep in np.unique(episodes)]}
    return {'groups': values, 'coordinate_normalized_mse': normalized.mean(0).tolist(), 'episode_ids': np.unique(episodes).tolist()}

def tests():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(80, 6)); y = x @ rng.normal(size=(6, 8)) + 2
    assert np.max(np.abs(predict(fit(x, y, 0), x) - y)) < 1e-10
    episodes = np.repeat(np.arange(8), 10); mask = episodes < 5
    m, decision = train_only(x, y, episodes, mask)
    xx, yy = x.copy(), y.copy(); xx[~mask] = 1e9; yy[~mask] = -1e9
    mm, dd = train_only(xx, yy, episodes, mask)
    assert decision == dd and all(np.array_equal(m[k], mm[k]) for k in m)
    folds = decision['fold_episode_ids']; assert sorted(v for fold in folds for v in fold) == list(range(5))
    torch.manual_seed(7); z = torch.randn(2, 48, 1, 4, 4).to(torch.bfloat16)
    assert torch.equal(Adapter('mlp').eval()(z), z)
    score = metrics(np.zeros((8, 8)), np.ones((8, 8)), np.ones(8), np.repeat([3, 9], 4))
    assert score['episode_ids'] == [3, 9] and len(score['groups']['all']['episode_normalized_mse']) == 2
    return {'passed': True, 'checks': ['ridge_exact_linear_fixture', 'heldout_mutation_no_effect_on_fit_or_selection', 'disjoint_episode_folds', 'zero_adapter_identity_bf16', 'episode_aggregation']}

def main():
    started = time.monotonic(); torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    assert not (OUT / 'summary.json').exists()
    freeze = json.loads((OUT / 'source-manifest.json').read_text())
    for name, sha in freeze['inputs'].items():
        assert digest(Path(name)) == sha, name
    assert json.loads((PRE / 'validation.json').read_text())['passed']
    rows = json.loads((PRE / 'sample-manifest.json').read_text())
    labels = np.load(PRE / 'aligned-state-targets.npz')
    state, target = labels['state'], labels['target']
    episodes = np.array([r['episode'] for r in rows])
    tr = np.array([r['split'] == 'train' for r in rows]); va = ~tr
    assert tr.sum() == 420 and va.sum() == 60
    cache = torch.load(ROOT / 'results/robustness-20260921/adapter-cache.pt', map_location='cpu', weights_only=False)
    clean = cache['clean'][[r['clean_index'] for r in rows]]
    visual = pooled(clean)
    joined = np.concatenate([visual, state], 1)
    readout, decision = train_only(joined, target, episodes, tr)
    proprio, pdecision = train_only(state, target, episodes, tr)
    write('selection.json', {'selected_at': datetime.now().astimezone().isoformat(), 'visual_proprio': decision, 'proprio_only': pdecision, 'selection_uses_only_training_episodes': True})
    np.savez(OUT / 'readout-coefficients.npz', **{'visual_' + k: v for k, v in readout.items()}, **{'proprio_' + k: v for k, v in proprio.items()})
    vrows = [r for r in rows if r['split'] == 'val']
    noisy = cache['noisy'][[r['noisy_indices'][0] for r in vrows]]
    cleanval = clean[torch.from_numpy(va)]
    y, stateval, ep = target[va], state[va], episodes[va]
    pred = {'proprio_only': predict(proprio, stateval)}
    stats = {}
    def record(name, z):
        pred[name] = predict(readout, np.concatenate([pooled(z), stateval], 1))
        stats[name] = {'mse_to_clean_latent': float((z.float() - cleanval.float()).square().mean())}
    record('identity_clean', cleanval); record('identity_noisy', noisy)
    previous = ROOT / 'results/robustness-20260921'
    assert digest(previous / 'adapter-selected.pt') == digest(previous / 'adapter-mlp-42.pt')
    for seed in [42, 43, 44]:
        ck = torch.load(previous / f'adapter-mlp-{seed}.pt', map_location='cpu', weights_only=True)
        model = Adapter(ck['kind']).eval(); model.load_state_dict(ck['state_dict'])
        for condition, z in [('clean', cleanval), ('noisy', noisy)]:
            with torch.inference_mode():
                restored = torch.cat([model(part) for part in z.split(8)])
            record(f'adapter{seed}_{condition}', restored)
    # Episode-cycle mismatch retains sampled-frame ordinal; never an independent trial.
    mapping = {int(e): int(np.unique(ep)[(i + 1) % len(np.unique(ep))]) for i, e in enumerate(np.unique(ep))}
    perm = np.empty(len(ep), dtype=int)
    for e, other in mapping.items():
        source, dest = np.where(ep == e)[0], np.where(ep == other)[0]
        assert len(source) == len(dest) == 12
        perm[source] = dest
    record('visual_mismatch_clean', cleanval[perm])
    results = {name: metrics(p, y, readout['ys'], ep) for name, p in pred.items()}
    bootstrap = np.random.default_rng(SEED).integers(0, 5, size=(2000, 5))
    paired = {}
    for name in pred:
        if name.startswith('adapter'):
            ref = 'identity_noisy' if name.endswith('noisy') else 'identity_clean'
            paired[name] = {}
            for group in ['all', 'translation', 'gripper']:
                base = np.array(results[ref]['groups'][group]['episode_normalized_mse'])
                delta = np.array(results[name]['groups'][group]['episode_normalized_mse']) - base
                paired[name][group] = {'normalized_mse_delta': float(delta.mean()), 'relative_change_percent': float(100 * delta.mean() / base.mean()), 'episode_paired_delta': delta.tolist(), 'episode_bootstrap_percentile95': np.quantile(delta[bootstrap].mean(1), [.025, .975]).tolist()}
    np.savez(OUT / 'predictions.npz', **pred, target=y, episode=ep)
    unchanged = all(digest(Path(name)) == sha for name, sha in freeze['inputs'].items())
    assert unchanged and all(np.isfinite(p).all() for p in pred.values())
    summary = {'completed_at': datetime.now().astimezone().isoformat(), 'scope': 'One-task cached validation development diagnostic, not independent confirmation or policy evaluation', 'train_frames': 420, 'validation_frames': 60, 'validation_episodes': np.unique(ep).tolist(), 'primary_adapter_seed': 42, 'adapter_seeds_all_reported': [42, 43, 44], 'results': results, 'paired_vs_identity': paired, 'latent_statistics': stats, 'selection': {'visual_proprio_alpha': decision['alpha'], 'proprio_alpha': pdecision['alpha']}, 'elapsed_seconds': time.monotonic() - started, 'inputs_unchanged': unchanged, 'new_policy_rollouts': 0, 'new_adapter_fits': 0, 'full_readouts_fitted': 2, 'limits': ['Validation used for prior adapter selection; development reuse only.', 'Pretrained representation may have seen demonstrations.', 'CPU FP32 adapter with BF16 input/output; GPU parity not tested.', 'State changes are achieved outcomes, not commands/contact/rotation.', 'One task, five evaluation episodes, one cached noise draw per frame.', 'Readout gains do not establish closed-loop benefit.']}
    write('summary.json', summary)
    print(json.dumps({'completed_at': summary['completed_at'], 'elapsed_seconds': summary['elapsed_seconds'], 'selection': summary['selection'], 'primary_noisy': paired['adapter42_noisy'], 'primary_clean': paired['adapter42_clean'], 'all_conditions_retained': list(results)}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(2)
    if args.check:
        print(json.dumps(tests(), indent=2))
    else:
        main()
