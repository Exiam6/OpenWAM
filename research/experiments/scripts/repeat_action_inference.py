"""54 preregistered chunks split across two fresh full-policy server processes."""
import argparse,hashlib,json,logging,sys,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from norm_common import intervene
from robustness_common import Adapter
from repeat_common import payload_hash,array_hash,traced_generate,fresh_payload
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/repeatability-20260921'
sys.path.insert(0,str(ROOT/'OpenWAM'))
from openwam.deploy.server import build_server_from_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--process',choices=['A','B'],required=True);args=p.parse_args()
    out=OUT/f'process-{args.process}';out.mkdir(exist_ok=True);assert not (out/'records.json').exists()
    capture=json.loads((OUT/'capture-summary.json').read_text());assert capture['all_initial_hashes_match']
    inputs=sorted(capture['records'],key=lambda r:(r['condition']=='noise',r['seed']))
    payloads={r['name']:json.loads((OUT/'inputs'/f"{r['name']}.json").read_text())for r in inputs}
    assert all(payload_hash(payloads[r['name']])==r['payload_sha256']for r in inputs)
    logging.basicConfig(level=logging.INFO);torch.manual_seed(42);torch.set_num_threads(4)
    cfg=OmegaConf.load(ROOT/'OpenWAM/configs/deploy.yaml');cfg.optimization.compile.enabled=False
    server=build_server_from_config(cfg,str(ROOT/'assets/dinov3-policy'),device='cuda:0')
    server._init_policy();assert not server._policy._async
    OmegaConf.save(server.cfg,out/'effective-config.yaml')
    encoder=server.engine.architecture.video_backbone.video_encoder
    weights=load_file(ROOT/'assets/dinov3-study/encoder.safetensors');prefix='video_backbone.video_encoder.'
    subset={k[len(prefix):]:v for k,v in weights.items()if k.startswith(prefix)}
    assert set(subset)==set(encoder.state_dict())
    assert all(torch.equal(encoder.state_dict()[k].cpu(),v.to(encoder.state_dict()[k].dtype))for k,v in subset.items())
    ckpath=ROOT/'results/robustness-20260921/adapter-selected.pt'
    assert hashlib.sha256(ckpath.read_bytes()).hexdigest()=='f0d0cf8eeae4b87e3d16a742c8733e05b82ec737a445dec6cd18efedd3092586'
    ck=torch.load(ckpath,map_location='cpu',weights_only=True);adapter=Adapter(ck['kind']).cuda().eval().requires_grad_(False);adapter.load_state_dict(ck['state_dict'],strict=True)
    assert encoder._out_norm.eps==1e-6 and not encoder._out_norm.elementwise_affine
    active={'arm':'identity'};latent=[];chunks=[];original=encoder.batch_encode
    def encode(video):
        z=original(video);value=intervene(z,active['arm'],adapter,encoder._apply_feature_norm)
        latent.append({'input_hash':array_hash(z.float().cpu().numpy()),'output_hash':array_hash(value.float().cpu().numpy()),'max_change':float((value-z).abs().max()),'dtype':str(z.dtype),'future_slots_exact':torch.equal(value[:,:,1:],z[:,:,1:])})
        return value
    encoder.batch_encode=encode;server.engine.generate=traced_generate(server.engine.generate,chunks)
    plan=[('identity',2)]if args.process=='A' else [('identity',1),('renorm',2),('adapter',2),('adapter_renorm',2)]
    records=[]
    try:
        for arm,reps in plan:
            active['arm']=arm
            for rep in range(reps):
                for row in inputs:
                    chunks.clear();latent.clear();server.reset();t=time.monotonic()
                    response=server.predict(fresh_payload(payloads[row['name']]));torch.cuda.synchronize()
                    assert len(chunks)==1 and len(latent)==1
                    chunk=chunks[0];first=np.array(response['action'],dtype=np.float32)
                    assert chunk.ndim==2 and chunk.shape[1]==20 and first.shape==(20,)
                    assert np.isfinite(chunk).all()and np.isfinite(first).all()
                    name=f"{arm}-r{rep}-{row['name']}";np.savez(out/f'{name}.npz',chunk=chunk,first_action=first)
                    rec={'name':name,'arm':arm,'rep':rep,'input':row['name'],'process':args.process,'seconds':time.monotonic()-t,'payload_sha256':row['payload_sha256'],'chunk_hash':array_hash(chunk),'first_hash':array_hash(first),'chunk_shape':list(chunk.shape),'first_action':first.tolist(),'latent':latent[0],'inference_horizon':server._policy._executor.inference_horizon,'binary_command_dims':list(getattr(server.engine.architecture,'binary_command_dims',())or())}
                    records.append(rec);(out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
                    assert payload_hash(payloads[row['name']])==row['payload_sha256']
                    print('REPEAT_CHUNK',json.dumps({k:rec[k]for k in ('name','seconds','chunk_hash','first_hash')}),flush=True)
        assert len(records)==(12 if args.process=='A' else 42)
        (out/'complete.json').write_text(json.dumps({'count':len(records),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'encoder_weights_exact':True,'process':args.process},indent=2)+'\n')
    finally:server.shutdown()
if __name__=='__main__':main()
