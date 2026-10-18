"""Gated native image/cache agreement after a fresh smoke-checkpoint deployment load."""
import argparse
import datetime as dt
import json
from pathlib import Path
import signal
import time

ROOT = Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
BASE = ROOT.parent / 'rae-factorial-20261010'


def admission():
    from build_cache import sha
    plan=json.loads((Path(__file__).parent/'plan.json').read_text())
    assert plan['scope']=='diagnostic_only_no_release' and plan['seconds_limit']==600
    for path,digest in plan['frozen_files'].items():assert sha(path)==digest,path
    assert sha(__file__)==plan['diagnostic_code_sha256']
    assert not (ROOT/'full-training-release.json').exists()
    assert json.loads((ROOT/'dedicated-preflight-run.json').read_text())['status']=='failed'
    p=json.loads((ROOT/'gpu-registration.json').read_text())
    for path in p['pause_paths']:assert not Path(path).exists()
    smoke=json.loads((ROOT/'training-smoke/vae48_native-seed42/result.json').read_text())
    assert smoke['complete'] and smoke['steps']==3
    assert sha(smoke['checkpoint'])==smoke['checkpoint_sha256']
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(plan['deadline'])
    return p,smoke,dict(remaining_seconds=590)


def main():
    p, smoke, budget = admission()
    import numpy as np
    import torch
    from PIL import Image
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.denoise_schedule import make_schedule
    from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder
    from build_cache import sha

    out = Path(__file__).parent / 'observations'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(diagnostic_only=True, full_training_released=False, original_gate_status='failed', completed=False, rows=[], selection='First registered training window of adjust_bottle, the task reached before the original failure')
    allowance = budget['remaining_seconds']

    def guard():
        assert time.monotonic() - start < allowance - 60
        assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(p['deadline'])
        for name in p['pause_paths']:
            assert not Path(name).exists()

    def stop(*_):
        raise TimeoutError('Deployment contract cap or termination')

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(max(1, int(allowance - 60)))
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        protocol = json.loads((BASE / 'protocol.json').read_text())
        audit = json.loads((ROOT / 'attempt-data-restored/input-audit.json').read_text())
        manifest_path = ROOT / 'attempt-data-restored/window-manifest.json'
        assert audit['passed'] and sha(manifest_path) == audit['window_manifest_sha256']
        manifest = json.loads(manifest_path.read_text())
        sources = {x['episode']: x for x in manifest['sources']}
        cache = torch.load(ROOT / 'vae-cache/cache.pt', map_location='cpu', weights_only=False, mmap=True)
        _, model = load_from_checkpoint_dir(str(Path(smoke['checkpoint']).parent), device='cuda:0', ckpt_name=Path(smoke['checkpoint']).name)
        encoder = model.video_backbone.video_encoder
        assert isinstance(encoder, WanVideoVAEEncoder) and model.video_backbone.vae is None
        assert model.action_dim == 20 and not model.binary_command_dims and model.normalizer is not None
        prompt_cache = {}
        captures={};current_label=['none'];original_forward=model.forward
        def capture_forward(*args,**kwargs):
            label=current_label[0]
            if label not in captures:
                captured={k:v.detach().cpu().clone() for k,v in kwargs.items() if torch.is_tensor(v) and k in ['latents','first_frame_latents','context','seq_lens','proprio','timestep']}
                for index,value in enumerate(args):
                    if torch.is_tensor(value):captured['argument_'+str(index)]=value.detach().cpu().clone()
                captured['flags']={k:kwargs.get(k) for k in ['fuse_vae_embedding_in_latents','num_clean_prefix_frames']}
                captures[label]=captured
            return original_forward(*args,**kwargs)
        model.forward=capture_forward
        for task in protocol['tasks'][:1]:
            guard()
            m = next(x for x in manifest['windows'] if x['task'] == task and x['split'] == 'train')
            i = m['cache_index']
            assert all(cache['metadata'][i][k] == m[k] for k in cache['metadata'][i])
            source = sources[m['episode']]
            assert sha(source['path']) == source['sha256']
            with np.load(source['path'], allow_pickle=False) as data:
                frame = Image.fromarray(data['images'][m['rgb_indices'][0]])
                raw_state = data['proprio'][m['start']].copy()
            proprio = model.normalize_deploy_proprio(raw_state).reshape(1, 20)
            proprio_difference = float((proprio - cache['proprio'][i:i+1]).abs().max())
            assert proprio_difference <= 1e-6
            with torch.no_grad():
                native_first = encoder.batch_encode(encoder.preprocess_video([frame]))
            cached_first = cache['vae'][i:i+1, :, :1].to('cuda', torch.bfloat16)
            assert native_first.shape == cached_first.shape == (1, 48, 1, 24, 20)
            latent_difference = float((native_first - cached_first).abs().max())
            assert latent_difference <= 1e-6

            def generate(use_image,label):
                current_label[0]=label
                guard()
                schedule = make_schedule('sync', video_scheduler=model.video_scheduler, action_scheduler=model.action_scheduler, num_steps=10, shift=5., shift_video=5.)
                kwargs = dict(first_frame_image=frame) if use_image else dict(first_frame_image=None, first_frame_latents=cached_first, fuse_vae_embedding_in_latents=True, num_clean_prefix_frames=0)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    actions = model.generate(schedule, protocol['prompts'][task], num_frames=9, action_num_frames=33, height=384, width=320,
                        seed=9210180+i, tiled=False, num_inference_steps=10, shift=5., decode_video=False,
                        prompt_embed_cache=prompt_cache, proprio=proprio, **kwargs)['actions']
                actions = np.asarray(actions)
                assert actions.shape == (32, 20) and np.isfinite(actions).all()
                return actions

            image_actions, cached_actions, repeated = generate(True,'image'), generate(False,'cached'), generate(True,'repeated')
            action_difference = float(np.max(np.abs(image_actions-cached_actions)))
            repeat_difference = float(np.max(np.abs(image_actions-repeated)))
            # Preserve all observations before evaluating any tolerance; this diagnostic cannot release training.
            captured_path=out/f'{task}-forward-inputs.pt';torch.save(captures,captured_path)
            input_differences={}
            for label in ['cached','repeated']:
                diff={}
                for key,value in captures['image'].items():
                    other=captures[label].get(key)
                    if torch.is_tensor(value) and torch.is_tensor(other):
                        diff[key]=dict(shape=list(value.shape),dtype=str(value.dtype),other_dtype=str(other.dtype),max_abs=float((value.float()-other.float()).abs().max()),exact=torch.equal(value,other))
                    else:diff[key]=dict(image=value,other=other)
                input_differences[label]=diff
            result['input_differences']=input_differences
            result['captured_inputs']=str(captured_path);result['captured_inputs_sha256']=sha(captured_path)
            result['original_tolerance_would_pass']=action_difference<=1e-5 and repeat_difference<=1e-5
            trace = out / f'{task}-actions.npz'
            np.savez_compressed(trace, image=image_actions, cached=cached_actions, repeated=repeated)
            result['rows'].append(dict(task=task, cache_index=i, source_sha256=source['sha256'], proprio_difference=proprio_difference,
                latent_difference=latent_difference, action_difference=action_difference, repeat_difference=repeat_difference,
                trace=str(trace), trace_sha256=sha(trace)))
        result.update(completed=True,checkpoint_sha256=smoke['checkpoint_sha256'],native_deployment_reload=True,
            policy_success_claim=False,notes='Diagnostic observations only. The original deployment gate remains failed regardless of these values; no full training, optimizer update, tiny fit, evaluation or new release is authorized.')
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        signal.alarm(0)
        result['occupied_seconds'] = time.monotonic() - start
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-admission', action='store_true')
    args = parser.parse_args()
    if args.check_admission:
        try:
            admission()
            print('admitted')
        except AssertionError as error:
            print(str(error))
            raise SystemExit(2)
    else:
        main()
