"""One seed's immutable pilot or remaining VAE evaluation under a shared3h cap."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from build_cache import sha
from stage_runner import run_child

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
BASE=ROOT.parent/'rae-factorial-20261010';R=ROOT.parent;SOURCE=Path(__file__).parent


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def admission(seed,stage):
    registration=ROOT/'gpu-registration.json';assert registration.exists(),'CPU-only: no GPU registration'
    p=json.loads(registration.read_text());assert p['gpu_launch_permitted'] and 'evaluation' in p['released_stages']
    assert p['evaluation_seconds_per_seed']==10800 and p['evaluation_pilot_seconds']==3600
    for file,digest in p['evaluation_files'].items():assert sha(file)==digest,file
    assert str(Path(__file__).resolve()) in p['evaluation_files']
    for path in p['pause_paths']:assert not Path(path).exists()
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
    trained=json.loads((ROOT/'runs'/f'vae48_native-seed{seed}'/'result.json').read_text())
    assert trained['complete'] and trained['steps']==20000 and sha(trained['checkpoint'])==trained['checkpoint_sha256']
    limit=3600
    if stage=='full':
        release=json.loads((ROOT/f'evaluation-release-seed{seed}.json').read_text())
        pilot_path=ROOT/'evaluation'/f'seed{seed}'/'pilot/result.json';pilot=json.loads(pilot_path.read_text())
        assert pilot['status']=='complete' and sha(pilot_path)==release['pilot_result_sha256']
        assert release['passed'] and release['budget_projection_passed']
        review=json.loads(Path(release['visual_review']).read_text())
        assert sha(release['visual_review'])==release['visual_review_sha256'] and review['visually_inspected']
        assert review['paired_videos']==15 and set(review['tasks'])==set(p['scenes'])
        budget=json.loads((ROOT/f'evaluation-budget-seed{seed}.json').read_text())
        assert budget['pilot_result_sha256']==sha(pilot_path)
        assert budget['prior_occupied_seconds']>=pilot['occupied_seconds']
        limit=budget['remaining_seconds'];assert 60<limit<=10800-budget['prior_occupied_seconds']
    return p,trained,limit


def stop(child):
    if child is None:return
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=15)


def main(seed,stage,gpu,allocation_end):
    p,trained,limit=admission(seed,stage)
    from evaluation_contract import validate
    end=min(dt.datetime.fromisoformat(allocation_end),dt.datetime.fromisoformat(p['deadline']))
    now=dt.datetime.now(dt.timezone.utc)
    if gpu not in [1,2,3,4]:
        assert dt.datetime.fromisoformat(p['extra_window_start'])<=now
        assert end<=dt.datetime.fromisoformat(p['extra_window_end'])
    assert (end-now).total_seconds()>limit+60
    assert os.environ['CUDA_VISIBLE_DEVICES']==p['gpus'][str(gpu)]
    occupants=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
    assert all(line.split(',')[0].strip()!=p['gpus'][str(gpu)] for line in occupants.splitlines() if line.strip()),'Evaluation requires a dedicated idle GPU'
    out=ROOT/'evaluation'/f'seed{seed}'/stage;out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();server=None
    record=dict(status='starting',seed=seed,stage=stage,gpu=gpu,pid=os.getpid(),started=now.isoformat(),groups=[],children=[],limit_seconds=limit,checkpoint=trained['checkpoint'],checkpoint_sha256=trained['checkpoint_sha256'])
    def save():write(out/'result.json',dict(record,occupied_seconds=time.monotonic()-start))
    def health():return 'Policy server exited' if server is not None and server.poll() is not None else None
    def guard():
        assert time.monotonic()-start<limit-60
        assert dt.datetime.now(dt.timezone.utc)<end-dt.timedelta(seconds=60)
        for pause in p['pause_paths']:assert not Path(pause).exists()
        assert health() is None,health()
    def terminated(*_):raise RuntimeError('Evaluation budget or termination')
    signal.signal(signal.SIGTERM,terminated);signal.signal(signal.SIGINT,terminated);signal.signal(signal.SIGALRM,terminated);signal.alarm(max(1,int(limit-45)));save()
    try:
        code=out/'code';shutil.copytree(ROOT/'evaluation-template',code);(code/'scenes').symlink_to(BASE/'scenes',target_is_directory=True)
        code_hashes={str(x):sha(x) for x in code.iterdir() if x.is_file()};write(out/'code-freeze.json',code_hashes)
        env=json.loads((ROOT/'base-environment.json').read_text());env.update(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],EXPECTED_RENDER_PCI=os.environ['EXPECTED_RENDER_PCI'],
            PATH=f'{R}/policy-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin',PYTHONPATH='/home/zifanz4/openwam-rae-research',ROBOT_CONFIG_ROOT=str(ROOT/'robot-configs'))
        simenv=dict(env,PATH=f'{R}/benchmark-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin',ROBOTWIN_RUNTIME_ROOT=str(out/'simulator-runtime'));simenv.pop('PYTHONPATH',None)
        write(out/'environment.json',env);port=23000+gpu*10
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        command=[str(R/'policy-env/bin/python'),'-u',str(code/'serve_policy.py'),'--checkpoint',trained['checkpoint'],'--port',str(port),'--log',str(out/'episode-rng.jsonl')]
        with (out/'server.log').open('x') as log:server=subprocess.Popen(command,env=env,cwd='/home/zifanz4/openwam-rae-research',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(server_pid=server.pid,server_command=command,status='loading_policy');save();loading=time.monotonic()
        while 'ENDTOEND_SERVER_READY' not in (out/'server.log').read_text(errors='replace'):
            guard();assert time.monotonic()-loading<900;time.sleep(2)
        def execute(name,command,dest,cap=600):
            guard();record['status']=name;save();seconds=min(cap,limit-(time.monotonic()-start)-30);assert seconds>30
            run=run_child(command,log=dest,seconds=seconds,deadline=end,pause_paths=p['pause_paths'],env=simenv,cwd='/home/zifanz4/openwam-rae-research',health_check=health)
            record['children'].append(dict(name=name,**run));save();assert run['complete'] and not run.get('error'),run
        smoke=out/'smoke';smoke.mkdir()
        execute('training_scene_smoke',[str(R/'benchmark-env/bin/python'),'-u',str(code/'simulation_smoke.py'),'--port',str(port),'--output',str(smoke)],smoke/'run.log')
        assert json.loads((smoke/'result.json').read_text())['passed']
        write(code/f'simulation-preflight-vae48_native-{seed}.json',dict(passed=True,checkpoint=trained['checkpoint'],code_sha256=code_hashes,result=str(smoke/'result.json')))
        reference_path=out.parent/'reference-actions.json'
        if stage=='pilot':
            execute('open_loop_references',[str(R/'benchmark-env/bin/python'),'-u',str(SOURCE/'reference_actions.py'),'--seed',str(seed),'--port',str(port),'--output',str(reference_path)],out/'references.log',600)
        references=json.loads(reference_path.read_text());assert references['passed'] and references['checkpoint_sha256']==trained['checkpoint_sha256'] and len(references['rows'])==60
        if stage=='full':
            prior=json.loads((out.parent/'pilot/result.json').read_text());assert prior['reference_sha256']==sha(reference_path)
        record['reference_sha256']=sha(reference_path);save()
        historical={(x['training_seed'],x['task'],x['scene_seed']):x for x in json.loads((ROOT/'historical-outcomes.json').read_text()) if x['route']=='raw768_native'}
        for task,all_scenes in p['scenes'].items():
            scenes=all_scenes[:5] if stage=='pilot' else all_scenes[5:]
            dest=out/'episodes'/task;dest.mkdir(parents=True)
            command=[str(R/'benchmark-env/bin/python'),'-u',str(code/'evaluate_policy.py'),'--scene-seeds',json.dumps(scenes),'--task',task,'--route','vae48_native','--seed',str(seed),'--condition','clean','--port',str(port),'--output',str(dest)]
            execute('closed_loop',command,dest/'run.log',len(scenes)*600+180)
            summary=json.loads((dest/'summary.json').read_text());assert summary['complete'] and summary['technical_failures']==0 and summary['episodes']==len(scenes)
            audits=[validate(dest/f'seed-{scene}'/'result.json',historical,references['rows']) for scene in scenes]
            write(dest/'independent-audit.json',dict(passed=True,rows=audits));record['groups'].append(dict(task=task,episodes=len(scenes),successes=summary['successes'],seconds=summary['seconds'],audit=str(dest/'independent-audit.json')));save()
        record.update(status='complete',complete=True,episodes=sum(x['episodes'] for x in record['groups']))
    except BaseException as error:record.update(status='failed',complete=False,error=repr(error));raise
    finally:
        signal.alarm(0);stop(server);record['finished']=dt.datetime.now(dt.timezone.utc).isoformat();save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,43,44],required=True);parser.add_argument('--stage',choices=['pilot','full'],required=True);parser.add_argument('--gpu',type=int,choices=range(8));parser.add_argument('--allocation-end');parser.add_argument('--check-admission',action='store_true');args=parser.parse_args()
    if args.check_admission:
        try:admission(args.seed,args.stage);print('admitted')
        except AssertionError as error:print(str(error));raise SystemExit(2)
    else:
        assert args.gpu is not None and args.allocation_end;main(args.seed,args.stage,args.gpu,args.allocation_end)
