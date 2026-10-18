"""Record native first chunks from frozen scene observations before scored rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
BASE=ROOT.parent/'rae-factorial-20261010'


def main(args):
    from build_cache import sha
    registration=ROOT/'gpu-registration.json';assert registration.exists(),'CPU-only: no GPU registration'
    p=json.loads(registration.read_text());assert p['gpu_launch_permitted'] and 'evaluation' in p['released_stages']
    for name,digest in p['evaluation_files'].items():assert sha(name)==digest,name
    assert str(Path(__file__).resolve()) in p['evaluation_files']
    trained=json.loads((ROOT/'runs'/f'vae48_native-seed{args.seed}'/'result.json').read_text());assert trained['complete'] and trained['steps']==20000
    assert sha(trained['checkpoint'])==trained['checkpoint_sha256']
    import numpy as np
    from PIL import Image
    sys.path.insert(0,'/home/zifanz4/openwam-runtime/assets-source/temporal-20260926/OpenWAM/benchmarks/robotwin')
    import openwam2robotwin_interface as interface
    protocol=json.loads((ROOT/'evaluation-template/eval-protocol.json').read_text())
    output=Path(args.output);partial=output.with_suffix('.partial.json')
    assert not output.exists() and not partial.exists(),'Preserve previous reference attempt'
    model=interface.ModelClient(host='127.0.0.1',port=args.port,send_state=True,state_dim=20,request_timeout=120,action_type='ee')
    current=dict(step=0,seed=0)
    def send(payload):
        payload=dict(payload);payload.update(_study_step=current['step'],_study_action_seed=current['seed'])
        return model._client.predict_once(payload)
    model._client.predict=send;rows={}
    def save_partial(error=None):
        value=dict(passed=False,training_seed=args.seed,checkpoint_sha256=trained['checkpoint_sha256'],rows=rows,error=error)
        temporary=partial.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(partial)
    save_partial()
    try:
        for task,scenes in p['scenes'].items():
            for scene in scenes:
                source=BASE/'scenes'/task/f'seed-{scene}';initial=json.loads((source/'initial-replay.json').read_text())
                state=np.asarray(initial['state'],dtype=np.float32)
                frames={name:np.asarray(Image.open(source/f'initial-{camera}.png')) for name,camera in [('head','head_camera'),('left','left_camera'),('right','right_camera')]}
                example=dict(cams=frames,lang=protocol['instructions'][task],state=state)
                action_seed=int(np.random.SeedSequence([scene,args.seed,811]).generate_state(1,dtype=np.uint32)[0]);current['seed']=action_seed
                copies=[]
                for _ in range(2):
                    interface.reset_model(model);actions=[]
                    for step in range(8):
                        current['step']=step;action=np.asarray(model.step(example));assert action.shape==(20,) and np.isfinite(action).all();actions.append(action.copy())
                    copies.append(np.stack(actions))
                assert np.array_equal(*copies),'Native replay changed after reset'
                files=[source/'initial-replay.json',*(source/f'initial-{camera}.png' for camera in ['head_camera','left_camera','right_camera'])]
                rows[f'{task}|{scene}']=dict(task=task,scene_seed=scene,training_seed=args.seed,action_seed=action_seed,
                    initial_state=hashlib.sha256(state.tobytes()).hexdigest(),instruction=example['lang'],actions=copies[0].tolist(),repeat_exact=True,
                    source_sha256={str(x):sha(x) for x in files})
                save_partial()
        assert len(rows)==60
        output.write_text(json.dumps(dict(passed=True,training_seed=args.seed,checkpoint_sha256=trained['checkpoint_sha256'],rows=rows,
            scope='All60 frozen initial observations, two8-action open-loop repetitions each. No simulator stepping, outcome selection or future observations. This checks deterministic serving replay, not an independent policy implementation.'),indent=2)+'\n')
    except BaseException as error:save_partial(repr(error));raise
    finally:model._client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,43,44],required=True);parser.add_argument('--port',type=int,required=True);parser.add_argument('--output',required=True);main(parser.parse_args())
