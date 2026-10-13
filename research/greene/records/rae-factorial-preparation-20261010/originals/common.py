"""Private matched native-policy pilot; no change to the native loss/forward."""
import datetime
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
CP = Path(P['source_checkpoint'])
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def guard():
    for pause in P['pause_paths']:
        assert not Path(pause).exists(), pause
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(P['deadline'])


def load_prefix(module, prefix):
    target = module.state_dict()
    with safe_open(CP, framework='pt', device='cpu') as f:
        names = {k[len(prefix):] for k in f.keys() if k.startswith(prefix)}
        assert names == set(target), (prefix, names ^ set(target))
        with torch.no_grad():
            for name, value in target.items():
                source = f.get_tensor(prefix + name)
                assert source.shape == value.shape
                value.copy_(source)


def build(variant, seed, *, frozen=True):
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    torch.manual_seed(seed)
    _, model, cfg = build_architecture_from_ckpt_dir(str(ROOT / 'templates' / variant), weights_required=False)
    model.freeze_modules(list(cfg.model.freeze))
    if frozen:
        load_prefix(model.video_backbone.text_encoder, 'video_backbone.text_encoder.')
        enc = model.video_backbone.video_encoder
        load_prefix(enc._m, 'video_backbone.video_encoder._m.')
        if enc._svae is not None:
            load_prefix(enc._svae, 'video_backbone.video_encoder._svae.')
    return model, cfg


def activate(model):
    model.set_dtype_device(torch.bfloat16, torch.device('cuda'))
    # FP32 master parameters + BF16 autocast; frozen feature/text weights BF16.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.init_training_schedulers()
    model.set_training_runtime(use_gradient_checkpointing=True)
    model.train()
    model.freeze_modules(list(OmegaConf.load(ROOT / 'templates/raw768_native/config.yaml').model.freeze))


def inputs_for(cache, ids, variant, *, checkpoint=True):
    index = torch.as_tensor(ids, dtype=torch.long)
    key = 'svae' if variant == 'svae48_native' else 'raw'
    x = cache[key][index].to('cuda', torch.bfloat16)
    task_ids = cache['task_ids'][index]
    return dict(
        input_latents=x, first_frame_latents=x[:, :, :1],
        context=cache['context'][task_ids].to('cuda', torch.bfloat16),
        seq_lens=cache['seq_lens'][task_ids].to('cuda'),
        height=384, width=320, num_frames=9, num_clean_prefix_frames=0,
        fuse_vae_embedding_in_latents=True, cfg_scale=1, cfg_merge=False,
        tiled=False, latents=None, vace_context=None, clip_feature=None, y=None,
        actions=cache['actions'][index].to('cuda', torch.bfloat16),
        proprio=cache['proprio'][index].to('cuda', torch.bfloat16),
        proprio_mask=torch.ones(len(ids), 1, 20, dtype=torch.bool),
        action_is_pad=torch.zeros(len(ids), 32, 20, dtype=torch.bool, device='cuda'),
        video_is_pad=torch.zeros(len(ids), 2, dtype=torch.bool, device='cuda'),
        use_gradient_checkpointing=checkpoint,
        use_gradient_checkpointing_offload=False,
        max_timestep_boundary=1., min_timestep_boundary=0.,
    )


def paired_loss(model, inputs, seed):
    # Separate streams prevent the 48-vs-768 video tensor size from shifting
    # action Gaussian draws. Native compute_loss samples both timesteps on CPU.
    torch.manual_seed(seed)
    original = torch.randn_like
    def sample(x, *args, **kwargs):
        assert not args and not kwargs
        assert x.ndim in (3, 5), x.shape
        gen = torch.Generator(device=x.device).manual_seed(seed + (23 if x.ndim == 3 else 17))
        return torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    with patch.object(torch, 'randn_like', sample), torch.autocast('cuda', dtype=torch.bfloat16):
        result = model.compute_loss(**inputs, lambda_video=1., lambda_action=1.)
    assert torch.randn_like is original
    assert all(torch.isfinite(v).all() for v in result.values())
    return result


def dev_metrics(model, cache, variant):
    model.eval()
    rows = []
    with torch.no_grad():
        for i, meta in enumerate(cache['metadata']):
            if meta['split'] != 'dev':
                continue
            scores = []
            for draw in range(3):
                out = paired_loss(model, inputs_for(cache, [i], variant, checkpoint=False), 7000000 + i * 10 + draw)
                scores.append({k: float(v) for k, v in out.items()})
            rows.append(dict(meta, **{k: float(np.mean([s[k] for s in scores])) for k in scores[0]}))
    per_task = {task: {key: float(np.mean([r[key] for r in rows if r['task'] == task]))
                       for key in ['loss_action', 'loss_video']}
                for task in P['tasks']}
    return dict(rows=rows, per_task=per_task,
                task_balanced_action_loss=float(np.mean([v['loss_action'] for v in per_task.values()])),
                note='Held-aside trajectory flow loss; NOT generated-action MSE or closed-loop success. No checkpoint selection.')
