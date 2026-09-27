#!/usr/bin/env python3
"""Post-training paired-view diagnostic and compression batch-geometry control.

No new fitting, training, or coefficient selection. Original evaluations remain
as audit artifacts; summary consumes evaluation-matched.json after this passes.
"""
import gc
import json
import time
import numpy as np
import torch
import cv2
import h5py
from PIL import Image
import control_study as c

n, p = c.n, c.p


@torch.inference_mode()
def extract_views(data, task):
    folder = c.OUT / task
    path = folder / 'paired-camera-current.pt'
    if path.exists(): return torch.load(path, weights_only=False)
    p.ensure_gpu(); c.gpu_guard(); enc = p.build_encoder()
    test = p.ids(data, 'test').tolist()
    out = {'test_ids': test, 'head_reencoded': [], 'front_replacement': []}
    differences = []; start = time.perf_counter()
    for ix in test:
        meta = data['manifest'][ix]
        with h5py.File(p.ROOT / 'data' / meta['episode_file'], 'r') as f:
            images = {camera: Image.fromarray(cv2.imdecode(np.frombuffer(bytes(f[f'observation/{camera}/rgb'][meta['start_frame']]), np.uint8), cv2.IMREAD_COLOR)) for camera in [*p.CAMS, 'front_camera']}
        for key, head in [('head_reencoded', 'head_camera'), ('front_replacement', 'front_camera')]:
            selected = {camera: images[camera] for camera in p.CAMS}
            selected['head_camera'] = images[head]
            frame = p.assemble_multiview_layout(selected, p.CAMS, 384, 320)
            z = enc.batch_encode_pooled_for_svae_training(enc.preprocess_video([frame] * 9))[0, :, :1].cpu().half()
            out[key].append(z)
            if key == 'head_reencoded':
                differences.append(float((z.float() - data['features'][ix, :, :1].float()).abs().max()))
        if len(out['head_reencoded']) % 20 == 0:
            c.gpu_guard(); print('VIEWS', task, len(out['head_reencoded']), '/', len(test), flush=True)
    for key in ['head_reencoded', 'front_replacement']: out[key] = torch.stack(out[key])
    torch.save(out, path)
    p.write_json(folder / 'view-extraction.json', {'test_clips': len(test), 'encoder_frames_per_batch': 9,
        'head_reencoded_vs_cached_max_abs': max(differences), 'wall_seconds': time.perf_counter() - start,
        'camera_replacement': 'head_camera input replaced with same-state front_camera; wrists retained',
        'input_encoding': 'Official RoboTwin cv2 decoding convention; no extra BGR/RGB swap'})
    del enc; gc.collect(); torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def vectorize(current, model):
    out = []
    for batch in current.split(2):
        if model is None: lat = batch.cuda().float()
        else:
            # Match the original clean path B=2,T=3. S-VAE has no cross-frame
            # attention; duplicate future slots affect GEMM geometry only.
            z = batch.cuda().float().repeat(1, 1, 3, 1, 1)
            with torch.autocast('cuda', dtype=torch.bfloat16): lat = model.encode_mean(z)
        out.append(c.current_vector(lat).cpu().numpy())
    return np.concatenate(out)


def evaluate(data, pca, task, name, views, corrupt, checkpoint=None):
    folder = c.OUT / task / name
    path = folder / 'evaluation-matched.json'
    if path.exists(): return
    old = json.loads((folder / 'evaluation.json').read_text())
    vec = np.load(folder / 'vectors.npy'); state = data['state'].numpy()
    probe = n.fit_probe(np.concatenate([vec, state], 1), data)
    te = p.ids(data, 'test').numpy(); y = data['target'].numpy()[te]
    episodes = np.asarray([data['manifest'][i]['episode'] for i in te])
    assert views['test_ids'] == corrupt['test_ids'] == te.tolist()
    def score(v):
        pred = n.predict(probe, np.concatenate([v, state[te]], 1))
        err = ((pred - y) / probe['ys']) ** 2
        return {**p.probe_metrics(pred, y, probe['ys']), 'episode_ids': sorted(set(episodes.tolist())),
            'episode_losses': [float(err[episodes == ep].mean()) for ep in sorted(set(episodes))],
            'per_clip_losses': err.mean(1).tolist()}
    model = None
    if checkpoint:
        ck = torch.load(checkpoint, weights_only=True); model = p.make_svae(pca); model.load_state_dict(ck['state_dict']); model.eval()
    cached_current = data['features'][p.ids(data, 'test'), :, :1]
    same_input = vectorize(cached_current, model)
    diff = float(np.max(np.abs(same_input - vec[te])))
    assert np.allclose(same_input, vec[te], atol=1e-6, rtol=1e-6), ('Reducer batch geometry control failed', task, name, diff)
    # Characterize the original T=1 numerical discrepancy separately.
    with torch.inference_mode():
        subset = cached_current[:2].cuda().float()
        if model is None: single = c.current_vector(subset).cpu().numpy()
        else:
            with torch.autocast('cuda', dtype=torch.bfloat16): lat = model.encode_mean(subset)
            single = c.current_vector(lat).cpu().numpy()
    old['numeric_control'] = {'all_test_current_vector_max_abs_vs_clean': diff,
        'first_batch_T1_vs_T3_vector_max_abs': float(np.max(np.abs(single - same_input[:2]))),
        'conditioning_test_clips': len(te), 'compression_batch': 'B=2,T=3 with duplicated unobserved slots',
        'future_context_added': False}
    old['probes']['matched_clean'] = score(same_input)
    for variant in ['noise10', 'brightness06']:
        old['probes'][variant] = score(vectorize(corrupt[variant], model))
    for variant in ['head_reencoded', 'front_replacement']:
        old['probes'][variant] = score(vectorize(views[variant], model))
    old['camera_note'] = 'Fixed same-state camera-pair transfer; not a continuous viewpoint sweep or contact test'
    p.write_json(path, old)
    print('MATCHED', task, name, json.dumps({k: old['probes'][k]['normalized_mse'] for k in ['clean', 'noise10', 'head_reencoded', 'front_replacement']}), 'delta', diff, flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


def main():
    assert (c.OUT / 'complete.json').exists(), 'Finish original fixed study before this diagnostic'
    assert (c.OUT / 'selection.json').exists()
    start = time.perf_counter()
    for task in n.TASKS:
        data, pca = c.load_task(task)
        views = extract_views(data, task)
        corrupt = torch.load(n.RUNROOT / task / 'corruptions-matched-batch.pt', weights_only=False)
        evaluate(data, pca, task, 'raw', views, corrupt)
        for seed in n.SEEDS:
            for weight in c.LAMBDAS:
                c.gpu_guard(); name = c.tag(weight, seed)
                evaluate(data, pca, task, name, views, corrupt, c.OUT / task / name / 'final.pt')
        del data, pca, views, corrupt; gc.collect(); torch.cuda.empty_cache()
    p.write_json(c.OUT / 'robustness-complete.json', {'wall_seconds': time.perf_counter() - start,
        'new_training_runs': 0, 'coefficients_reselected': False, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('ROBUSTNESS COMPLETE', flush=True)


if __name__ == '__main__': main()
