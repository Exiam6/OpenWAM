"""CPU construction with explicit frozen assets and paired trainable initialization."""
import json
from pathlib import Path
import torch
from safetensors import safe_open

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
BASE=ROOT.parent/'rae-factorial-20261010'
PROTOCOL=json.loads((BASE/'protocol.json').read_text())

@torch.no_grad()
def copy_prefix(module, path, prefix=''):
    state=module.state_dict()
    with safe_open(str(path),framework='pt',device='cpu') as f:
        names={k[len(prefix):] for k in f.keys() if k.startswith(prefix)}
        assert names==set(state), (prefix,names^set(state))
        for name,target in state.items():
            value=f.get_tensor(prefix+name)
            assert value.shape==target.shape
            target.copy_(value)
            assert torch.equal(target.cpu(),value.to(target.dtype))
    return len(state)

@torch.no_grad()
def build_cpu(seed):
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder
    assert seed in [42,43,44]
    torch.set_num_threads(4);torch.manual_seed(seed)
    _,model,cfg=build_architecture_from_ckpt_dir(str(ROOT/'templates/vae48_native'),weights_required=False)
    model.freeze_modules(list(cfg.model.freeze))
    encoder=model.video_backbone.video_encoder
    assert isinstance(encoder,WanVideoVAEEncoder) and model.video_backbone.vae is None
    vae_count=copy_prefix(encoder._m,ROOT/'assets/wan-vae48.safetensors')
    text_count=copy_prefix(model.video_backbone.text_encoder,Path(PROTOCOL['source_checkpoint']),'video_backbone.text_encoder.')
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    init=BASE/'initialization'/f'common-seed{seed}.safetensors'
    matched=[];excluded=[]
    with safe_open(str(init),framework='pt',device='cpu') as f:
        shared=set(f.keys()); assert set(trainable)==shared
        for name,param in trainable.items():
            value=f.get_tensor(name)
            if param.shape!=value.shape:
                assert name.startswith(('video_backbone.dit.patch_embedding.','video_backbone.dit.head.head.')),name
                excluded.append(dict(name=name,raw_shape=list(value.shape),vae_shape=list(param.shape)))
            else:
                param.copy_(value);assert torch.equal(param.cpu(),value.to(param.dtype));matched.append(name)
    assert excluded and all(not p.requires_grad for p in encoder.parameters())
    assert all(not p.requires_grad for p in model.video_backbone.text_encoder.parameters())
    assert all(p.device.type=='cpu' for p in model.parameters())
    assert not torch.cuda.is_initialized()
    result=dict(seed=seed,common_initialization=str(init),matched_trainable_tensors=len(matched),dimension_specific_exclusions=excluded,
        frozen_vae_tensors=vae_count,frozen_text_tensors=text_count,trainable_parameters=sum(p.numel() for p in trainable.values()),
        total_parameters=sum(p.numel() for p in model.parameters()),trained_official_policy_weights_loaded=False,cuda_initialized=False)
    return model,cfg,result
