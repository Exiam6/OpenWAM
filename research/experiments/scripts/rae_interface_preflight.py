"""Training-only native Wan baseline interface check; no task scores or fits."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(os.environ['WAM_ROOT'])
RUN = ROOT / 'results/rae-baseline-20260922'
PROTOCOL = json.loads((RUN / 'interface-protocol.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(name, obj):
    dest = RUN / name
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(dest)


# Avoid unrelated policy initialization; import the unchanged native classes.
for name in ['openwam', 'openwam.model', 'openwam.model.video_backbone',
             'openwam.model.video_backbone.encoder', 'openwam.model.video_backbone.wan',
             'openwam.model.video_backbone.wan.models']:
    m = types.ModuleType(name)
    m.__path__ = [str(ROOT / 'OpenWAM' / Path(*name.split('.')))]
    sys.modules[name] = m
from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38


def main():
    assert not (RUN / 'result.json').exists(), 'Completed preflight must not be repeated'
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    fixture_path = Path(PROTOCOL['fixture']['path'])
    assert sha(fixture_path) == PROTOCOL['fixture']['sha256']
    weight_path = ROOT / 'assets/wan22-vae-baseline/Wan2.2_VAE.pth'
    assert weight_path.stat().st_size == PROTOCOL['asset']['size']
    assert sha(weight_path) == PROTOCOL['asset']['sha256']
    fixture = torch.load(fixture_path, map_location='cpu', weights_only=False)
    assert len(fixture['rows']) == 8
    assert all(x['split'] == 'train' for x in fixture['rows'])
    images = []
    for raw, row in zip(fixture['encoded_images'], fixture['rows']):
        raw = bytes(raw.numpy())
        assert hashlib.sha256(raw).hexdigest() == row['encoded_head_hashes'][0]
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        images.append(Image.fromarray(arr).resize((320, 384), Image.Resampling.BILINEAR))
    start = time.monotonic()
    state = torch.load(weight_path, map_location='cpu', weights_only=True, mmap=True)
    if 'model_state' in state:
        state = state['model_state']
    # This is the exact native converter: prefix every original weight with model.
    model = WanVideoVAE38()
    model.load_state_dict({'model.' + k: v for k, v in state.items()}, strict=True, assign=True)
    enc = WanVideoVAEEncoder(model.to(device='cuda', dtype=torch.bfloat16)).eval().requires_grad_(False)
    assert dataclasses.asdict(enc.properties)['z_dim'] == 48
    result = {'protocol_sha256': sha(RUN / 'interface-protocol.json'),
              'asset_sha256': sha(weight_path), 'properties': dataclasses.asdict(enc.properties),
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'model_load_seconds': time.monotonic() - start, 'operations': [],
              'training_rows': fixture['rows'], 'task_scores': None, 'passed': False}

    def encode(frames, name):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.monotonic()
        pix = enc.preprocess_video(frames)
        z = enc.batch_encode(pix)
        torch.cuda.synchronize()
        assert torch.isfinite(z).all()
        assert tuple(z.shape) == (1, 48, (len(frames) - 1) // 4 + 1, 24, 20)
        entry = {'name': name, 'seconds': time.monotonic() - begin,
                 'shape': list(z.shape), 'pixel_min': float(pix.min()), 'pixel_max': float(pix.max()),
                 'peak_allocated_GB': torch.cuda.max_memory_allocated() / 1e9,
                 'latent_mean': float(z.float().mean()), 'latent_std': float(z.float().std())}
        assert -1 <= entry['pixel_min'] <= entry['pixel_max'] <= 1
        result['operations'].append(entry)
        write('progress.json', result)
        print(json.dumps(entry), flush=True)
        return z

    zs = []
    with torch.inference_mode():
        for i, image in enumerate(images):
            zs.append(encode([image], f'frame{i}').cpu())
        repeated = encode([images[0]] * 9, 'nine_repeats').cpu()
        inverted = Image.fromarray(255 - np.array(images[0]))
        changed = encode([images[0]] + [inverted] * 8, 'future_changed').cpu()
        replay = encode([images[0]], 'cache_reset').cpu()
        deltas = {'single_vs_video': float((zs[0] - repeated[:, :, :1]).abs().max()),
                  'future_leak': float((repeated[:, :, :1] - changed[:, :, :1]).abs().max()),
                  'cache_reset': float((zs[0] - replay).abs().max()),
                  'future_sensitivity': float((repeated[:, :, 1:] - changed[:, :, 1:]).abs().max())}
        result['causal_deltas'] = deltas
        write('progress.json', result)
        assert max(deltas[k] for k in ['single_vs_video', 'future_leak', 'cache_reset']) <= 1e-6, deltas
        assert deltas['future_sensitivity'] > 0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.monotonic()
        decoded = enc.decode(zs[0].cuda(), tiled=False)
        torch.cuda.synchronize()
        assert tuple(decoded.shape) == (1, 3, 1, 384, 320) and torch.isfinite(decoded).all()
        result['pixel_decode'] = {'shape': list(decoded.shape), 'seconds': time.monotonic() - begin,
                                  'peak_allocated_GB': torch.cuda.max_memory_allocated() / 1e9,
                                  'training_frame_MSE_minus1_plus1': float((decoded.float() - enc.preprocess_video([images[0]]).float()).square().mean())}
        latents = torch.cat(zs).float()
        normalized = F.layer_norm(latents.permute(0, 2, 3, 4, 1), (48,), eps=1e-6).permute(0, 4, 1, 2, 3)
        raw_pool = F.adaptive_avg_pool2d(latents[:, :, 0], (2, 2)).flatten(1)
        ln_pool = F.adaptive_avg_pool2d(normalized[:, :, 0], (2, 2)).flatten(1)
        result['readout_geometry'] = {'wan_and_svae_pooled_dim': 192, 'raw_dino_pooled_dim': 3072,
                                     'native_pooled_norms': raw_pool.norm(dim=1).tolist(),
                                     'layernorm_pooled_norms': ln_pool.norm(dim=1).tolist(),
                                     'existing_DINO_shape': list(fixture['features']['L12'].shape)}
        torch.save({'wan_latents': latents, 'rows': fixture['rows'], 'protocol_sha256': result['protocol_sha256']}, RUN / 'training-fixture-wan.pt')
    result['passed'] = True
    result['total_seconds'] = time.monotonic() - start
    write('result.json', result)
    print('PREFLIGHT PASSED', flush=True)


if __name__ == '__main__':
    main()
