"""CPU-only Raw/VAE paired evidence for every completed pilot scene."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
G=ROOT.parent/'ex4-g_calibratedfactorial_1017'
CAMERAS=['head_camera','left_camera','right_camera']


from build_cache import sha


def readiness(seed):
    path=ROOT/'gpu-registration.json';assert path.exists(),'CPU-only: VAE evaluation not registered'
    from corrected_protocol import load_registration
    p=load_registration();assert p['gpu_launch_permitted'] and 'evaluation' in p['released_stages']
    for file,digest in p['evaluation_files'].items():assert sha(file)==digest,file
    pilot=ROOT/'evaluation'/f'seed{seed}'/'pilot'
    result=json.loads((pilot/'result.json').read_text());assert result['status']=='complete' and result['episodes']==15
    baseline=json.loads((G/'frozen-reuse.json').read_text());assert baseline['complete']
    rows={tuple(x['identity'][1:]):x for x in baseline['rows'] if x['identity'][0]=='raw768_native'}
    pairs=[]
    for task,scenes in p['scenes'].items():
        audit=json.loads((pilot/'episodes'/task/'independent-audit.json').read_text());assert audit['passed'] and len(audit['rows'])==5
        for scene in scenes[:5]:
            new_path=pilot/'episodes'/task/f'seed-{scene}'/'result.json';new=json.loads(new_path.read_text())
            audited=next(x for x in audit['rows'] if x['identity']==['vae48_native',seed,task,scene])
            assert audited['verified'] and audited['result']==new and audited['result_sha256']==sha(new_path)
            assert audited['trace_sha256']==sha(new_path.parent/'trace.jsonl')
            assert [new['route'],new['training_seed'],new['task'],new['scene_seed']]==['vae48_native',seed,task,scene]
            old=rows[(seed,task,scene)];old_path=Path(old['result_path'])
            assert old['verified'] and sha(old_path)==old['result_sha256']
            assert sha(old_path.parent/'trace.jsonl')==old['trace_sha256']
            assert old['result']['initial_state']==new['initial_state'] and new['status']=='complete'
            pairs.append((old_path,new_path))
    assert len(pairs)==15
    return p,pairs


def make(old_path,new_path):
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    old,new=json.loads(old_path.read_text()),json.loads(new_path.read_text())
    seed,task,scene=new['training_seed'],new['task'],new['scene_seed']
    out=ROOT/'paired-pilot-videos'/f'seed{seed}'/task;out.mkdir(parents=True,exist_ok=True)
    video=out/f'scene-{scene}.mp4';metadata=video.with_suffix('.json')
    if metadata.exists():
        saved=json.loads(metadata.read_text())
        assert saved['raw_result_sha256']==sha(old_path) and saved['vae_result_sha256']==sha(new_path)
        assert saved['full_decode_passed'] and saved['video_sha256']==sha(video)
        assert sha(saved['preview'])==saved['preview_sha256']
        return saved
    assert not video.exists(),'Partial video preserved; no automatic retry'
    directories=[old_path.parent/'frames',new_path.parent/'frames'];indices=[]
    for directory in directories:
        values=sorted(int(x.name.split('-')[0]) for x in directory.glob('*-head_camera.jpg'))
        assert values and values==list(range(values[-1]+1))
        result=old if directory==directories[0] else new
        assert values[-1]==result['steps']
        assert all((directory/f'{i:04d}-{cam}.jpg').exists() for i in values for cam in CAMERAS)
        indices.append(values)
    with Image.open(directories[0]/'0000-head_camera.jpg') as first:width,height=first.size
    size=(width*3,(height+36)*2);count=max(map(len,indices));samples=sorted(set([0,count//5,2*count//5,3*count//5,4*count//5,count-1]))
    binary=imageio_ffmpeg.get_ffmpeg_exe();command=[binary,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','12','-i','-','-an','-c:v','libx264','-threads','2','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    child=subprocess.Popen(command,stdin=subprocess.PIPE);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24);previews=[]
    try:
        for index in range(count):
            canvas=Image.new('RGB',size,'black');draw=ImageDraw.Draw(canvas)
            for row,(label,directory) in enumerate(zip(['Raw','VAE'],directories)):
                top=row*(height+36);draw.text((10,top+4),label,font=font,fill='white');step=min(index,indices[row][-1])
                for column,camera in enumerate(CAMERAS):
                    with Image.open(directory/f'{step:04d}-{camera}.jpg') as frame:
                        assert frame.size==(width,height);canvas.paste(frame,(column*width,top+36))
            child.stdin.write(canvas.tobytes())
            if index in samples:previews.append(canvas.resize((480,276)))
        child.stdin.close();assert child.wait(timeout=60)==0
    except BaseException:
        child.terminate()
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        raise
    subprocess.run([binary,'-v','error','-i',str(video),'-f','null','-'],check=True,timeout=300)
    sheet=Image.new('RGB',(960,828),'white')
    for index,preview in enumerate(previews):sheet.paste(preview,((index%2)*480,(index//2)*276))
    preview=video.with_suffix('.jpg');sheet.save(preview,quality=92)
    record=dict(seed=seed,task=task,scene_seed=scene,raw_success=old['success'],vae_success=new['success'],raw_result=str(old_path),vae_result=str(new_path),
        raw_result_sha256=sha(old_path),vae_result_sha256=sha(new_path),video=str(video),video_sha256=sha(video),preview=str(preview),preview_sha256=sha(preview),
        frames=count,raw_frames=len(indices[0]),vae_frames=len(indices[1]),fps=12,preview_indices=samples,camera_order=CAMERAS,full_decode_passed=True,
        alignment='Action index, not physical time; shorter episode holds its final frame.',
        scope='Raw DINO and Wan VAE under the same corrected executor, same seed and scene, fixedstep20000. Three actual policy cameras; no external whole-robot camera. Not the old/fixed calibration comparison.')
    metadata.write_text(json.dumps(record,indent=2)+'\n');return record


def main(seed,check):
    _,pairs=readiness(seed)
    if check:print(json.dumps(dict(ready=True,pairs=len(pairs))));return
    records=[make(*pair) for pair in pairs]
    path=ROOT/f'pilot-media-seed{seed}.json';assert not path.exists(),'Preserve previous complete media manifest'
    path.write_text(json.dumps(dict(seed=seed,paired_videos=15,records=records),indent=2)+'\n');print(json.dumps(dict(seed=seed,paired_videos=15,full_decode_passed=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,43,44],required=True);parser.add_argument('--check-readiness',action='store_true');args=parser.parse_args()
    try:main(args.seed,args.check_readiness)
    except AssertionError as error:
        if args.check_readiness:print(str(error));raise SystemExit(2)
        raise
