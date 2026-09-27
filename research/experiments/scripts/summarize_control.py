#!/usr/bin/env python3
"""Summarize every fixed-budget control-supervised run; no best-seed filtering."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/control-20260921'
OUT = ROOT / 'public-results/control-20260921'
OUT.mkdir(parents=True, exist_ok=True)
TASKS = ['adjust_bottle', 'handover_block', 'place_object_basket']
SEEDS = [42, 43, 44]
WEIGHTS = [0., .1, 1.]
VARIANTS = ['clean', 'noise10', 'brightness06', 'shuffled_visual', 'head_reencoded', 'front_replacement']

def read(p): return json.loads(p.read_text())
def tag(w, s): return f'aux{w:g}-s{s}'
def stat(a):
    return {'mean': float(np.mean(a)), 'std_across_training_seeds': float(np.std(a, ddof=1)) if len(a) > 1 else None, 'values': a}

def paired(base, candidate):
    # First average over all fixed training seeds, then resample episodes.
    a, b = np.asarray(base).mean(0), np.asarray(candidate).mean(0)
    diff = b - a
    rng = np.random.default_rng(20260921)
    means = rng.choice(diff, (10000, len(diff)), replace=True).mean(1)
    return {'candidate_minus_baseline': float(diff.mean()), 'relative_change_percent': float(100 * (b.mean() / a.mean() - 1)),
            'episode_paired_bootstrap_95ci': np.quantile(means, [.025, .975]).tolist(),
            'interval_scope': 'Conditional on three fixed training seeds; episodes are the resampling unit'}

complete = read(RUN / 'complete.json')
selection = read(RUN / 'selection.json')
robustness = read(RUN / 'robustness-complete.json')
s = {'status': 'completed_exploratory_offline_study', 'protocol': read(RUN / 'protocol.json'),
     'checks': read(RUN / 'checks.json'), 'robustness_completion': robustness, 'selection': selection, 'tasks': {}, 'resources': {**complete, 'training_seconds': 0., 'peak_allocated_gb': 0.},
     'limitations': ['Repeated development episode split; fresh episodes required for confirmation',
                     'Future achieved-state readouts, not action commands or closed-loop success',
                     'Raw and compact readouts differ in input dimension and capacity',
                     'Fixed paired-camera transfer only; no continuous viewpoint sweep or verified contact label',
                     'Camera-transfer errors also reflect a head-view-trained readout; not a pure test of encoder information loss',
                     'Fixed final 2000-step budget, no convergence claim',
                     'Separate compressor trained per task; no shared multitask-compressor validation',
                     'New compressors require latent alignment or policy adaptation before deployment']}
for task in TASKS:
    folder = RUN / task
    rows = {tag(w, seed): read(folder / tag(w, seed) / 'evaluation-matched.json') for w in WEIGHTS for seed in SEEDS}
    raw = read(folder / 'raw/evaluation-matched.json')
    block = {'view_extraction': read(folder / 'view-extraction.json'), 'raw': raw, 'methods': {}, 'paired_vs_zero': {}, 'proprio_only': raw['proprio_only']}
    for w in WEIGHTS:
        group = [rows[tag(w, seed)] for seed in SEEDS]
        block['methods'][f'aux{w:g}'] = {'validation_mse': stat([r['validation_mse'] for r in group]),
            'numeric_controls': [r['numeric_control'] for r in group],
            'front_vs_head_error_ratio': stat([r['probes']['front_replacement']['normalized_mse']/r['probes']['head_reencoded']['normalized_mse'] for r in group]),
            'probes': {v: stat([r['probes'][v]['normalized_mse'] for r in group]) for v in VARIANTS},
            'translation_rmse_mm': {v: stat([r['probes'][v]['translation_rmse_mm'] for r in group]) for v in VARIANTS},
            'gripper_rmse': {v: stat([r['probes'][v]['gripper_rmse'] for r in group]) for v in VARIANTS},
            'reconstruction': {v: stat([r['reconstruction'][v] for r in group]) for v in ['all_mse', 'condition_mse', 'pooled_target_mse']}}
        if w != 0:
            block['paired_vs_zero'][f'aux{w:g}'] = {}
            for v in VARIANTS:
                a = [rows[tag(0, seed)]['probes'][v]['episode_losses'] for seed in SEEDS]
                b = [rows[tag(w, seed)]['probes'][v]['episode_losses'] for seed in SEEDS]
                for seed in SEEDS:
                    assert rows[tag(0, seed)]['probes'][v]['episode_ids'] == rows[tag(w, seed)]['probes'][v]['episode_ids']
                block['paired_vs_zero'][f'aux{w:g}'][v] = paired(a, b)
    s['tasks'][task] = block
    for w in WEIGHTS:
        for seed in SEEDS:
            tr = read(folder / tag(w, seed) / 'training.json')
            s['resources']['training_seconds'] += tr['wall_seconds']
            s['resources']['peak_allocated_gb'] = max(s['resources']['peak_allocated_gb'], tr['peak_allocated_gb'])
            if w == 0:
                s['checks'].setdefault('old_vs_rerun_baseline_max_abs', {})[task + f'-s{seed}'] = tr['previous_baseline_weight_max_abs']
s['macro_relative_changes'] = {f'aux{w:g}': {v: float(np.mean([s['tasks'][t]['paired_vs_zero'][f'aux{w:g}'][v]['relative_change_percent'] for t in TASKS])) for v in VARIANTS} for w in [.1, 1.]}
(OUT / 'summary.json').write_text(json.dumps(s, indent=2, allow_nan=False))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
colors = ['#547257', '#7c8baf', '#a9773c']
for col, (task, label) in enumerate(zip(TASKS, ['Adjust bottle', 'Handover block', 'Object to basket'])):
    block = s['tasks'][task]
    for row, variant in enumerate(['clean', 'noise10', 'front_replacement']):
        ax = axes[row, col]
        for i, w in enumerate(WEIGHTS):
            v = block['methods'][f'aux{w:g}']['probes'][variant]
            ax.bar(i, v['mean'], yerr=v['std_across_training_seeds'], color=colors[i], capsize=4)
        ax.axhline(block['raw']['probes'][variant]['normalized_mse'], color='#777777', linestyle='--', linewidth=1, label='Raw DINO (larger probe input)')
        ax.set_xticks(range(3), ['Original', 'Aux 0.1', 'Aux 1'])
        ax.set_title(label + [' / clean', ' / noise sigma 10/255', ' / head-to-front camera'][row])
        ax.set_ylabel('Independent probe MSE'); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
axes[0, 0].legend(frameon=False, fontsize=7)
fig.suptitle('Task-supervised compression / 3 seeds / fixed 2,000 steps\nError bars: training-seed SD; lower is better', fontsize=12)
fig.savefig(OUT / 'probe-results.svg'); fig.savefig(OUT / 'probe-results.png', dpi=180)
print(json.dumps({'selection': selection['selected_weight'], 'macro': s['macro_relative_changes'], 'resources': s['resources']}, indent=2))
