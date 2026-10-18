"""CPU-only matched VAE media, reusing reviewed pilot clips by exact hashes."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
G=ROOT.parent/'ex4-g_calibratedfactorial_1017';SOURCE=Path(__file__).parent
COMPARISONS={'primary_raw_minus_vae':('raw768_native',['Raw','VAE']),'secondary_compact_minus_vae':('svae48_native',['Compact','VAE'])}


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def available():
    rows={}
    for path in [G/'frozen-reuse.json',*G.glob('runs/*/evaluation/fixed/*/independent-audit.json'),*ROOT.glob('evaluation/seed*/*/episodes/*/independent-audit.json')]:
        audit=json.loads(path.read_text());assert audit.get('passed',audit.get('complete',False))
        for row in audit['rows']:
            identity=tuple(row['identity'])
            if identity[0] not in ['raw768_native','svae48_native','vae48_native']:continue
            assert row['verified'] and identity not in rows;rows[identity]=row
    return rows


def specs(p,rows):
    def item(row):return {k:row[k] for k in ['result_path','result_sha256','trace_sha256','result']}
    result=[]
    for comparison,(baseline,labels) in COMPARISONS.items():
        for seed in p['seeds']:
            for task,scenes in p['scenes'].items():
                for scene in scenes:
                    left,right=(baseline,seed,task,scene),('vae48_native',seed,task,scene)
                    if left not in rows or right not in rows:continue
                    key=f'{comparison}/seed{seed}/{task}/scene-{scene}'
                    spec=dict(key=key,kind='model',comparison=comparison,seed=seed,task=task,scene_seed=scene,labels=labels,routes=[baseline,'vae48_native'],left=item(rows[left]),right=item(rows[right]),output=str(ROOT/'full-media'/f'{key}.json'))
                    if comparison=='primary_raw_minus_vae' and scene in scenes[:5]:
                        # Let the frozen pilot path produce its complete evidence before adopting it.
                        manifest=ROOT/f'pilot-media-seed{seed}.json'
                        if not manifest.exists():continue
                        data=json.loads(manifest.read_text());assert data['seed']==seed and data['paired_videos']==15
                        existing=next(x for x in data['records'] if x['task']==task and x['scene_seed']==scene)
                        spec['reuse_metadata']=str(Path(existing['video']).with_suffix('.json'))
                    result.append(spec)
    return result


def stop(child):
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=5)


def main():
    p=json.loads((ROOT/'gpu-registration.json').read_text())
    assert any(ROOT.glob('evaluation/seed*/pilot/episodes/*/independent-audit.json')),'Wait for the first actual audited VAE pilot group'
    root=ROOT/'full-media';root.mkdir(exist_ok=False)
    state=dict(pid=os.getpid(),status='starting',planned=360,created=dt.datetime.now(dt.timezone.utc).isoformat());active={};seen=set();records=[];last_refresh=0.;pending=[]
    def save():
        state.update(time=dt.datetime.now(dt.timezone.utc).isoformat(),completed=len(records),active={k:dict(pid=v[0].pid,started_monotonic=v[1]) for k,v in active.items()});write(root/'queue.json',state)
    def terminated(*_):raise RuntimeError('CPU media queue termination')
    signal.signal(signal.SIGTERM,terminated);signal.signal(signal.SIGINT,terminated);save()
    try:
        while len(records)<360:
            assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
            assert not any(Path(x).exists() for x in p['pause_paths'])
            for key,(child,start,spec) in list(active.items()):
                code=child.poll()
                if code is None:assert time.monotonic()-start<180;continue
                assert code==0,(key,code)
                row=json.loads(Path(spec['output']).read_text());assert row['key']==key and row['full_decode_passed'];records.append(row);del active[key]
            if time.monotonic()-last_refresh>30:
                rows=available();pending=specs(p,rows);last_refresh=time.monotonic();state.update(available_vae_rows=sum(k[0]=='vae48_native' for k in rows),available_pairs=len(pending))
            for spec in [x for x in pending if x['key'] not in seen][:max(0,2-len(active))]:
                key=spec['key'];path=root/'specs'/f'{key}.json';path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('x') as f:json.dump(spec,f,indent=2)
                log=root/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                with log.open('x') as f:child=subprocess.Popen([sys.executable,'-u',str(SOURCE/'paired_video.py'),'--spec',str(path)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                active[key]=(child,time.monotonic(),spec);seen.add(key)
            state['status']='encoding' if active else 'waiting_audited_results';save()
            if len(records)<360:time.sleep(1 if active else 15)
        assert len({x['key'] for x in records})==360
        write(root/'manifest.json',dict(complete=True,paired_videos=360,records=records,scope='180Raw/VAE plus180Compact/VAE pairs; not additional evaluations.'))
        state['status']='complete';save()
    except BaseException as error:state.update(status='failed',error=repr(error));save();raise
    finally:
        for child,_,_ in active.values():stop(child)
        state['finished']=dt.datetime.now(dt.timezone.utc).isoformat();save()


if __name__=='__main__':main()
