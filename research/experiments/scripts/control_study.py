#!/usr/bin/env python3
"""Bounded task-supervised compression study. No policy-success claims."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import night as n

p = n.p
torch.use_deterministic_algorithms(True)
OUT = p.ROOT / 'results/control-20260921'
LAMBDAS = [0., .1, 1.]
STEPS = 2000


def tag(weight, seed):
    return f'aux{weight:g}-s{seed}'


def current_vector(mu):
    cur = mu[:, :, 0].float()
    cur = F.layer_norm(cur.permute(0, 2, 3, 1), (cur.shape[1],), eps=1e-6).permute(0, 3, 1, 2)
    b, c, h, w = cur.shape
    assert h % 2 == 0 and w % 2 == 0
    # Same four non-overlapping pooling cells; deterministic backward.
    return cur.reshape(b, c, 2, h // 2, 2, w // 2).mean((3, 5)).flatten(1)


def train_stats(data):
    ids = p.ids(data, 'train')
    return {key: {'mean': data[key][ids].mean(0), 'std': data[key][ids].std(0, unbiased=False).clamp_min(floor)}
            for key, floor in [('state', 1e-5), ('target', 1e-6)]}


def make_head(state_dim, seed):
    # Auxiliary initialization must not advance the compressor's Gaussian RNG.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed + 100000)
        return nn.Sequential(nn.Linear(192 + state_dim, 128), nn.GELU(), nn.Linear(128, 8)).cuda()


def auxiliary_loss(head, mu, state, target, stats):
    state = (state - stats['state']['mean']) / stats['state']['std']
    target = (target - stats['target']['mean']) / stats['target']['std']
    prediction = head(torch.cat([current_vector(mu), state], dim=1))
    return F.mse_loss(prediction.float(), target.float())


def combined_loss(base, auxiliary, weight):
    # Avoid changing bf16 gradient accumulation order through a zero path.
    return base if weight == 0 else base + weight * auxiliary


def gpu_guard():
    uid = os.environ['CUDA_VISIBLE_DEVICES']
    output = subprocess.check_output(['nvidia-smi', '-i', uid, '--query-gpu=ecc.errors.uncorrected.volatile.total', '--format=csv,noheader,nounits'], text=True).strip()
    assert output == '0', ('GPU ECC error', output)
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True).splitlines()
    foreign = [r for r in rows if r.split(',')[0].strip() == uid and int(r.split(',')[1].strip()) != os.getpid()]
    assert not foreign, ('Device claimed by another process; stopping our run', foreign)


def checks(data, pca, dest):
    dest.mkdir(parents=True, exist_ok=True)
    x = data['features'][:2].cuda().float()
    stat = {key: {k: v.cuda() for k, v in row.items()} for key, row in train_stats(data).items()}
    torch.manual_seed(4242)
    model = p.make_svae(pca)
    head = make_head(data['state'].shape[1], 42)
    state, target = data['state'][:2].cuda(), data['target'][:2].cuda()
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        a = model.encode_mean(x)
        changed = x.clone(); changed[:, :, 1:] = changed[:, :, 1:].flip(2) + 123.
        b = model.encode_mean(changed)
    assert torch.equal(a[:, :, :1], b[:, :, :1]), 'Future features leaked into observed latent'
    normalized = F.layer_norm(a[:, :, 0].float().permute(0, 2, 3, 1), (a.shape[1],), eps=1e-6).permute(0, 3, 1, 2)
    reference_pool = F.adaptive_avg_pool2d(normalized, (2, 2)).flatten(1)
    assert torch.allclose(current_vector(a), reference_pool, atol=1e-6, rtol=1e-6)
    mu = a.detach().float().requires_grad_()
    auxiliary_loss(head, mu, state, target, stat).backward()
    assert mu.grad[:, :, :1].abs().sum() > 0
    assert mu.grad[:, :, 1:].abs().sum() == 0
    # Identical stochastic forward state; adding a zero-weight auxiliary term
    # must reproduce all original compressor gradients exactly.
    model.train(); rng = torch.cuda.get_rng_state()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(x); base, _ = p.svae_loss(out, beta=1e-5, cos_weight=1.)
    base.backward(); grads = {k: v.grad.detach().clone() for k, v in model.named_parameters()}
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True); torch.cuda.set_rng_state(rng)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(x); base, _ = p.svae_loss(out, beta=1e-5, cos_weight=1.)
        aux = auxiliary_loss(head, out['mu'], state, target, stat)
    combined_loss(base, aux, 0.).backward()
    difference = max(float((v.grad - grads[k]).abs().max()) for k, v in model.named_parameters())
    assert difference == 0., difference
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(x); aux = auxiliary_loss(head, out['mu'], state, target, stat)
    aux.backward()
    assert model.enc_proj.weight.grad.abs().sum() > 0
    assert model.dec_proj.weight.grad is None
    # Statistics must ignore every held-out label and state.
    altered = dict(data)
    for key in ['state', 'target']:
        altered[key] = data[key].clone()
        altered[key][p.ids(data, 'test')] += 999.
    before, after = train_stats(data), train_stats(altered)
    assert all(torch.equal(before[key][k], after[key][k]) for key in before for k in before[key])
    result = {'future_input_invariance': True, 'deterministic_pool_matches_reference_tolerance_1e6': True, 'auxiliary_gradient_current_only': True,
              'lambda_zero_gradient_max_abs': difference, 'auxiliary_reaches_encoder_not_decoder': True,
              'train_statistics_ignore_test': True}
    p.write_json(dest / 'checks.json', result)
    print('CHECKS', json.dumps(result), flush=True)
    del model, head, x, out, grads; gc.collect(); torch.cuda.empty_cache()


def train(data, pca, weight, seed, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'final.pt'
    if path.exists():
        return path
    gpu_guard()
    torch.manual_seed(seed)
    model = p.make_svae(pca)
    head = make_head(data['state'].shape[1], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .99), weight_decay=1e-4)
    hopt = torch.optim.AdamW(head.parameters(), lr=1e-4, betas=(.9, .99), weight_decay=1e-4)
    stats = train_stats(data)
    stats_gpu = {key: {k: v.cuda() for k, v in row.items()} for key, row in stats.items()}
    rng = torch.Generator().manual_seed(seed)
    train_ids, val_ids = p.ids(data, 'train'), p.ids(data, 'val')
    x = data['features']; initial = n.reconstruction(model, x, val_ids)
    start = time.perf_counter(); history = []; torch.cuda.reset_peak_memory_stats()
    for step in range(1, STEPS + 1):
        selected = train_ids[torch.randint(len(train_ids), (2,), generator=rng)]
        z = x[selected].cuda().float(); state = data['state'][selected].cuda(); target = data['target'][selected].cuda()
        model.train(); head.train(); opt.zero_grad(set_to_none=True); hopt.zero_grad(set_to_none=True)
        lr = 1e-4 * min(1., step / 20) * (.95 + .05 * np.cos(np.pi * step / STEPS))
        for optimizer in [opt, hopt]:
            for group in optimizer.param_groups: group['lr'] = lr
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(z)
            base, _ = p.svae_loss(out, beta=1e-4 * min(1., step / (STEPS * .2)), cos_weight=1.)
            head_mu = out['mu'].detach() if weight == 0 else out['mu']
            aux = auxiliary_loss(head, head_mu, state, target, stats_gpu)
            loss = combined_loss(base, aux, weight)
        assert torch.isfinite(loss), (step, base, aux)
        loss.backward()
        if weight == 0:
            aux.backward()  # Train the reference head without changing compressor gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        opt.step(); hopt.step()
        if step % 200 == 0:
            gpu_guard()
            row = {'step': step, 'base_loss': float(base.detach()), 'auxiliary_loss': float(aux.detach()),
                   'val_reconstruction': n.reconstruction(model, x, val_ids), 'elapsed_seconds': time.perf_counter() - start}
            history.append(row); p.write_json(folder / 'curve.json', history)
            print('TRAIN', folder.parent.name, folder.name, json.dumps(row), flush=True)
    torch.save({'format_version': 2, 'model_config': model.config_dict(),
                'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'step': STEPS, 'seed': seed, 'auxiliary_weight': weight}, path)
    torch.save({'head_state_dict': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, 'stats': stats}, folder / 'auxiliary.pt')
    row = {'seed': seed, 'auxiliary_weight': weight, 'steps': STEPS, 'initial_validation': initial,
           'wall_seconds': time.perf_counter() - start, 'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
           'head_parameters': sum(v.numel() for v in head.parameters()), 'head_hidden': 128,
           'input_state_dim': data['state'].shape[1], 'target_dim': data['target'].shape[1]}
    if weight == 0:
        old = torch.load(n.RUNROOT / folder.parent.name / f'svae-w1-s{seed}' / 'final.pt', weights_only=True)
        row['previous_baseline_weight_max_abs'] = max(float((model.state_dict()[k].cpu() - v).abs().max()) for k, v in old['state_dict'].items())
    p.write_json(folder / 'training.json', row)
    del model, head, opt, hopt, out; gc.collect(); torch.cuda.empty_cache()
    return path


@torch.inference_mode()
def vectors(data, pca, name, checkpoint=None):
    model = None
    if checkpoint:
        ck = torch.load(checkpoint, weights_only=True)
        model = p.make_svae(pca); model.load_state_dict(ck['state_dict']); model.eval()
    out = []
    for batch in data['features'].split(2):
        if name == 'raw': lat = batch.cuda().float()
        else:
            with torch.autocast('cuda', dtype=torch.bfloat16): lat = model.encode_mean(batch.cuda().float())
        out.append(current_vector(lat).cpu().numpy())
    del model; gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(out)


def fit_validation(data, pca, task, name, checkpoint=None):
    folder = OUT / task / name
    folder.mkdir(parents=True, exist_ok=True)
    vecpath, path = folder / 'vectors.npy', folder / 'validation.json'
    if path.exists(): return json.loads(path.read_text())
    vec = vectors(data, pca, name, checkpoint)
    np.save(vecpath, vec)
    allvec = np.concatenate([vec, data['state'].numpy()], 1)
    probe = n.fit_probe(allvec, data)
    result = {'name': name, 'val_mse': float(probe['val_mse']), 'alpha': float(probe['alpha']), 'visual_dim': vec.shape[1]}
    p.write_json(path, result)
    print('VALIDATION', task, json.dumps(result), flush=True)
    return result


def evaluate(data, pca, task, name, checkpoint=None):
    folder = OUT / task / name; path = folder / 'evaluation.json'
    if path.exists(): return
    vec = np.load(folder / 'vectors.npy'); state = data['state'].numpy()
    allvec = np.concatenate([vec, state], 1); probe = n.fit_probe(allvec, data)
    te = p.ids(data, 'test').numpy(); y = data['target'].numpy()[te]
    episode_ids = np.asarray([data['manifest'][i]['episode'] for i in te])
    result = {'name': name, 'validation_mse': float(probe['val_mse']), 'alpha': float(probe['alpha']), 'probes': {}}
    def score(pred):
        errors = ((pred - y) / probe['ys']) ** 2
        return {**p.probe_metrics(pred, y, probe['ys']), 'episode_ids': sorted(set(episode_ids.tolist())),
                'episode_losses': [float(errors[episode_ids == e].mean()) for e in sorted(set(episode_ids))],
                'per_clip_losses': errors.mean(1).tolist()}
    result['probes']['clean'] = score(n.predict(probe, allvec[te]))
    # Task-local derangement, fixed identically for every representation.
    permutation = np.random.default_rng(20260921).permutation(len(te))
    permutation = np.roll(permutation, 1)[np.argsort(permutation)]
    assert not np.any(permutation == np.arange(len(te)))
    shuffled = np.concatenate([vec[te][permutation], state[te]], 1)
    result['probes']['shuffled_visual'] = score(n.predict(probe, shuffled))
    result['shuffle_note'] = 'Held-out within-task visual derangement; mismatched-pair dependence diagnostic, not deployment perturbation'
    model = None
    if checkpoint:
        ck = torch.load(checkpoint, weights_only=True); model = p.make_svae(pca); model.load_state_dict(ck['state_dict']); model.eval()
        result['reconstruction'] = n.reconstruction(model, data['features'], p.ids(data, 'test'))
    corrupt = torch.load(n.RUNROOT / task / 'corruptions-matched-batch.pt', weights_only=False)
    assert corrupt['test_ids'] == te.tolist()
    for variant in ['noise10', 'brightness06']:
        out = []
        with torch.inference_mode():
            for batch in corrupt[variant].split(2):
                if model is None: lat = batch.cuda().float()
                else:
                    with torch.autocast('cuda', dtype=torch.bfloat16): lat = model.encode_mean(batch.cuda().float())
                out.append(current_vector(lat).cpu().numpy())
        perturbed = np.concatenate([np.concatenate(out), state[te]], 1)
        result['probes'][variant] = score(n.predict(probe, perturbed))
    proprio = n.fit_probe(state, data)
    result['proprio_only'] = p.probe_metrics(n.predict(proprio, state[te]), y, proprio['ys'])
    result['auxiliary_head_used_for_metrics'] = False
    p.write_json(path, result)
    print('EVALUATION', task, name, json.dumps({k: v['normalized_mse'] for k, v in result['probes'].items()}), flush=True)
    del model, corrupt; gc.collect(); torch.cuda.empty_cache()


def load_task(task):
    folder = n.RUNROOT / task
    data = torch.load(folder / 'features.pt', map_location='cpu', weights_only=False, mmap=True)
    pca = torch.load(folder / 'pca.pt', weights_only=True)
    assert data['features'].shape == (600, 768, 3, 24, 20)
    assert len(set(m['episode'] for m in data['manifest'] if m['split'] == 'train')) == 35
    assert len(set(m['episode'] for m in data['manifest'] if m['split'] == 'val')) == 5
    assert len(set(m['episode'] for m in data['manifest'] if m['split'] == 'test')) == 10
    return data, pca


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--checks-only', action='store_true'); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); p.ensure_gpu(); gpu_guard(); start = time.perf_counter()
    data, pca = load_task(n.TASKS[0]); checks(data, pca, OUT)
    del data, pca; gc.collect(); torch.cuda.empty_cache()
    if args.checks_only: return
    protocol = {'tasks': n.TASKS, 'seeds': n.SEEDS, 'lambdas': LAMBDAS, 'steps': STEPS,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'selection': 'Global mean per-task/seed validation MSE ratio to lambda0; test locked until selection',
                'old_test_reused_for_development': True, 'confirmatory_new_episodes_required': True,
                'deterministic_algorithms': True, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'target': '4-raw-step achieved translation delta and future achieved gripper; not action commands'}
    p.write_json(OUT / 'protocol.json', protocol)
    validations = {}
    for task in n.TASKS:
        data, pca = load_task(task); validations[task] = {}
        validations[task]['raw'] = fit_validation(data, pca, task, 'raw')
        for seed in n.SEEDS:
            for weight in LAMBDAS:
                gpu_guard(); name = tag(weight, seed)
                checkpoint = train(data, pca, weight, seed, OUT / task / name)
                validations[task][name] = fit_validation(data, pca, task, name, checkpoint)
        del data, pca; gc.collect(); torch.cuda.empty_cache()
    scores = {str(weight): float(np.mean([validations[t][tag(weight, seed)]['val_mse'] / validations[t][tag(0, seed)]['val_mse'] for t in n.TASKS for seed in n.SEEDS])) for weight in LAMBDAS}
    selected = min(LAMBDAS, key=lambda weight: scores[str(weight)])
    p.write_json(OUT / 'selection.json', {'selected_weight': selected, 'validation_ratio_scores': scores, 'all_validation': validations, 'chosen_before_test_evaluation': True})
    print('SELECTION', selected, json.dumps(scores), flush=True)
    for task in n.TASKS:
        data, pca = load_task(task)
        evaluate(data, pca, task, 'raw')
        for seed in n.SEEDS:
            for weight in LAMBDAS:
                gpu_guard(); name = tag(weight, seed)
                evaluate(data, pca, task, name, OUT / task / name / 'final.pt')
        del data, pca; gc.collect(); torch.cuda.empty_cache()
    p.write_json(OUT / 'complete.json', {'wall_seconds': time.perf_counter() - start, 'training_runs': 27, 'selected_weight': selected, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('COMPLETE', flush=True)


if __name__ == '__main__': main()
