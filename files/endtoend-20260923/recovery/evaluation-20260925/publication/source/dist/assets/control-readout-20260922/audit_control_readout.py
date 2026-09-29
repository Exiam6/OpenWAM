"""Recompute reported errors from saved predictions, without fitting or inference."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path('/data02/zifanz4/openwam-experiments/results/control-readout-20260922')
PRE = Path('/data02/zifanz4/openwam-experiments/results/representation-preflight-20260922')

def main():
    source = json.loads((OUT / 'source-manifest.json').read_text())
    for path, expected in source['inputs'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    summary = json.loads((OUT / 'summary.json').read_text())
    predictions = np.load(OUT / 'predictions.npz')
    co = np.load(OUT / 'readout-coefficients.npz')
    rows = json.loads((PRE / 'sample-manifest.json').read_text())
    labels = np.load(PRE / 'aligned-state-targets.npz')
    train = np.array([r['split'] == 'train' for r in rows])
    assert np.array_equal(predictions['target'], labels['target'][~train])
    assert np.array_equal(predictions['episode'], [r['episode'] for r in rows if r['split'] == 'val'])
    scale = labels['target'][train].std(0).clip(1e-6)
    assert np.array_equal(co['visual_ys'], scale) and np.array_equal(co['proprio_ys'], scale)
    selection = json.loads((OUT / 'selection.json').read_text())
    assert source['frozen_at'] < selection['selected_at'] < summary['completed_at']
    assert (OUT / 'exit-code.txt').read_text().strip() == '0'
    expected = {'proprio_only', 'identity_clean', 'identity_noisy', 'visual_mismatch_clean'} | {f'adapter{s}_{c}' for s in [42, 43, 44] for c in ['clean', 'noisy']}
    assert set(summary['results']) == expected
    ep = predictions['episode']; unique = np.unique(ep)
    boot = np.random.default_rng(20260922).integers(0, 5, (2000, 5))
    for name in expected:
        error = (predictions[name] - predictions['target']) / scale
        for group, idx in [('all', np.arange(8)), ('translation', np.arange(6)), ('gripper', np.arange(6, 8))]:
            squared = np.square(error[:, idx])
            observed = summary['results'][name]['groups'][group]
            assert np.isclose(squared.mean(), observed['normalized_mse'], rtol=1e-12)
            per = np.array([squared[ep == e].mean() for e in unique])
            np.testing.assert_allclose(per, observed['episode_normalized_mse'], rtol=1e-12)
            if name.startswith('adapter'):
                baseline = 'identity_noisy' if name.endswith('noisy') else 'identity_clean'
                reference = np.square((predictions[baseline] - predictions['target'])[:, idx] / scale[idx])
                deltas = per - [reference[ep == e].mean() for e in unique]
                saved = summary['paired_vs_identity'][name][group]
                np.testing.assert_allclose(np.quantile(deltas[boot].mean(1), [.025, .975]), saved['episode_bootstrap_percentile95'], rtol=1e-10, atol=1e-14)
    result = {'passed': True, 'frozen_source_and_inputs_verified': len(source['inputs']), 'conditions_retained': len(expected), 'validation_episodes': len(unique), 'predictions_per_condition': len(ep), 'target_scaling_uses_training_only': True, 'all_group_metrics_recomputed': True, 'episode_bootstrap_recomputed': True, 'selection_before_scoring_verified': True, 'no_new_fit_or_inference': True}
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
