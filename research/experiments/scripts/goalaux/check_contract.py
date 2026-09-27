"""One bounded CPU contract check on an existing TRAIN-only fixture; no fit."""
import datetime
import hashlib
import importlib.util
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
import h5py
from objective import waypoint_objective, with_waypoint_objective

PRIVATE = Path('/home/zifanz4/openwam-experiments')
RUNTIME = Path('/data02/zifanz4/openwam-experiments')
STUDY = PRIVATE / 'studies/goalaux-20260922'
LAYER = RUNTIME / 'results/layers-20260922'
NATIVE = RUNTIME / 'OpenWAM/openwam/model/video_backbone/encoder/svae/model.py'
CHECKPOINT = LAYER / 'L12/seed42/final.pt'
FIXTURE = LAYER / 'transfer/parity-fixture.pt'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(name, obj):
    (STUDY / name).write_text(json.dumps(obj, indent=2) + '\n')


def main():
    assert not (STUDY / 'contract-started.json').exists(), 'No check replay'
    protocol = json.loads((STUDY / 'contract-protocol.json').read_text())
    now = datetime.datetime.now().astimezone()
    assert now < datetime.datetime.fromisoformat(protocol['deadline'])
    for base in [PRIVATE / 'studies/layers-20260922', PRIVATE, Path('/home/zifanz4/.local/state/openwam-selfcheck')]:
        assert not (base / 'paused.json').exists(), 'Paused'
    for name, expected in protocol['sha256'].items():
        assert sha(Path(name)) == expected, name
    assert torch.cuda.device_count() == 0, 'CPU-only contract'
    torch.set_num_threads(2)
    start = time.monotonic()
    dump('contract-started.json', {'time': now.isoformat(), 'cpu_threads': 2, 'gpus': 0})
    spec = importlib.util.spec_from_file_location('native_svae', NATIVE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    fixture = torch.load(FIXTURE, map_location='cpu', weights_only=False)
    chosen = next(i for i, row in enumerate(fixture['rows']) if row['split'] == 'train' and row['frame'] == 0)
    row = fixture['rows'][chosen]
    sample_key = (row['task'], row['episode'])
    train = json.loads((LAYER / 'initial-goal/training-samples.json').read_text())
    matches = [x for x in train if (x['task'], x['episode']) == sample_key and x['frame'] == 0]
    assert len(matches) == 1
    with h5py.File(RUNTIME / row['path'], 'r') as h5:
        raw = bytes(h5['observation/head_camera/rgb'][0])
        assert hashlib.sha256(raw).hexdigest() == row['encoded_head_hashes'][0]
    constants = json.loads((LAYER / 'initial-goal/constants.json').read_text())[row['task']]
    target = torch.tensor([(np.asarray(matches[0]['target_xy']) - constants['mean']) / constants['std']], dtype=torch.float32)
    x = fixture['features']['L12'][chosen:chosen+1].float().clone()
    del fixture
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    def model():
        m = native.SVAE(**checkpoint['model_config'])
        m.load_state_dict(checkpoint['state_dict'], strict=True)
        return m.train()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        heads = torch.nn.ModuleList([torch.nn.Linear(48 * 4, 2) for _ in range(3)])
    task = torch.tensor([['adjust_bottle', 'handover_block', 'place_object_basket'].index(row['task'])])
    states, gradients, losses, rngs = [], [], [], []
    for auxiliary_disabled in [False, True]:
        m = model()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(.9, .99), weight_decay=1e-4)
        torch.manual_seed(20260922)
        out = m(x)
        base, _ = native.svae_loss(out, beta=1e-4, cos_weight=1.)
        loss = with_waypoint_objective(base, out['mu'], None, None, None, 0.) if auxiliary_disabled else base
        assert loss is base
        rngs.append(torch.get_rng_state().clone())
        loss.backward()
        gradients.append({k: p.grad.clone() for k, p in m.named_parameters()})
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
        opt.step()
        states.append({k: v.clone() for k, v in m.state_dict().items()})
        losses.append(float(loss.detach()))
        del out, base, loss, opt, m
    assert losses[0] == losses[1]
    assert torch.equal(rngs[0], rngs[1])
    assert all(torch.equal(gradients[0][k], gradients[1][k]) for k in gradients[0])
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    gradient_tensors = len(gradients[0])
    del states, gradients
    print('ZERO_WEIGHT_EXACT', flush=True)
    m = model()
    out = m(x)
    aux = waypoint_objective(out['mu'], heads, task, target)
    aux.backward()
    encoder_grads = [p.grad for n, p in m.named_parameters() if n.startswith('enc_')]
    assert all(g is not None and torch.isfinite(g).all() for g in encoder_grads)
    encoder_norm = float(torch.sqrt(sum(g.square().sum() for g in encoder_grads)))
    assert encoder_norm > 0
    assert all(p.grad is None for n, p in m.named_parameters() if n.startswith('dec_'))
    active_head_norm = float(torch.sqrt(sum(p.grad.square().sum() for p in heads[int(task[0])].parameters())))
    assert active_head_norm > 0 and np.isfinite(active_head_norm)
    del out, aux
    m.zero_grad(set_to_none=True)
    m.eval()
    # Keep B,T and current frame identical; change only the future frame.
    later = x.flip(-1) * -3
    pair_a = torch.cat([x, x], dim=2)
    pair_b = torch.cat([x, later], dim=2)
    with torch.no_grad():
        mu_a = m.encode_mean(pair_a)[:, :, :1]
        mu_b = m.encode_mean(pair_b)[:, :, :1]
    assert torch.equal(mu_a, mu_b), 'Future frame changed first latent'
    pair_b.requires_grad_(True)
    causal_loss = waypoint_objective(m.encode_mean(pair_b), heads, task, target)
    causal_loss.backward()
    assert torch.count_nonzero(pair_b.grad[:, :, 1:]) == 0
    assert float(pair_b.grad[:, :, :1].norm()) > 0
    assert sha(CHECKPOINT) == protocol['sha256'][str(CHECKPOINT)]
    result = {'passed': True, 'finished_at': datetime.datetime.now().astimezone().isoformat(),
              'elapsed_seconds': time.monotonic()-start, 'peak_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'fixture': {'task': row['task'], 'episode': row['episode'], 'frame': 0, 'split': 'train', 'shape': list(x.shape)},
              'input_image_hash_matches_raw_hdf5': True, 'zero_weight_loss_rng_gradients_and_adamw_update_exact': True,
              'gradient_tensors_compared': gradient_tensors, 'auxiliary_encoder_gradient_norm': encoder_norm,
              'active_head_gradient_norm': active_head_norm, 'auxiliary_decoder_gradients_absent': True,
              'future_frame_change_leaves_initial_latent_exact': True, 'future_input_gradient_exact_zero': True,
              'checkpoint_unchanged': True, 'training_checkpoint_saved': False, 'fresh_or_development_scores_read': False,
              'limitations': ['One seed42 training sample; CPU fp32 only', 'No BF16/GPU update parity or throughput test',
                              'No performance gain measured; no new held-out data or compressor fit']}
    dump('contract-result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
