"""Encode one frozen CPU evidence pair, or adopt an already verified paired clip."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from build_cache import sha

CAMERAS=['head_camera','left_camera','right_camera']


def checked_pair(spec):
    rows=[]
    for index,name in enumerate(['left','right']):
        item=spec[name];path=Path(item['result_path']);row=json.loads(path.read_text())
        assert sha(path)==item['result_sha256'] and row==item['result']
        assert sha(path.parent/'trace.jsonl')==item['trace_sha256']
        assert row['status']=='complete' and type(row['success']) is bool
        assert row['route']==spec['routes'][index]
        assert (row['training_seed'],row['task'],row['scene_seed'])==(spec['seed'],spec['task'],spec['scene_seed'])
        rows.append(row)
    for key in ['initial_state','action_seed','step_limit','evaluation_instruction']:
        assert rows[0][key]==rows[1][key],key
    if spec['kind']=='calibration':assert rows[0]['route']==rows[1]['route']
    else:assert spec['kind']=='model' and rows[0]['route']!=rows[1]['route']
    return rows


def encode(spec):
    started=time.monotonic();rows=checked_pair(spec);output=Path(spec['output']);assert not output.exists()
    output.parent.mkdir(parents=True,exist_ok=True)
    if spec.get('reuse_metadata'):
        path=Path(spec['reuse_metadata']);old=json.loads(path.read_text())
        assert spec['labels']==['Raw','VAE'] and spec['routes']==['raw768_native','vae48_native']
        assert old['seed']==spec['seed'] and old['task']==spec['task'] and old['scene_seed']==spec['scene_seed']
        assert old['full_decode_passed'] and old['raw_result_sha256']==spec['left']['result_sha256'] and old['vae_result_sha256']==spec['right']['result_sha256']
        assert Path(old['raw_result'])==Path(spec['left']['result_path']) and Path(old['vae_result'])==Path(spec['right']['result_path'])
        video=Path(old['video']);preview=Path(old['preview'])
        assert sha(video)==old['video_sha256'] and sha(preview)==old['preview_sha256']
        count=old['frames'];samples=old['preview_indices'];counts=[old['raw_frames'],old['vae_frames']]
        assert counts==[x['steps']+1 for x in rows] and count==max(counts)
        reused=dict(metadata=str(path),sha256=sha(path))
    else:
        import imageio_ffmpeg
        from PIL import Image,ImageDraw,ImageFont
        directories=[Path(spec[x]['result_path']).parent/'frames' for x in ['left','right']];counts=[]
        for directory,row in zip(directories,rows):
            values=sorted(int(x.name.split('-')[0]) for x in directory.glob('*-head_camera.jpg'))
            assert values==list(range(row['steps']+1))
            assert all((directory/f'{step:04d}-{camera}.jpg').exists() for step in values for camera in CAMERAS)
            counts.append(len(values))
        with Image.open(directories[0]/'0000-head_camera.jpg') as frame:width,height=frame.size
        size=(width*3,(height+36)*2);count=max(counts);samples=sorted(set([0,count//5,2*count//5,3*count//5,4*count//5,count-1]))
        video=output.with_suffix('.mp4');preview=output.with_suffix('.jpg');assert not video.exists() and not preview.exists(),'Preserve partial encoding'
        binary=imageio_ffmpeg.get_ffmpeg_exe();command=[binary,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','12','-i','-','-an','-c:v','libx264','-threads','2','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        child=subprocess.Popen(command,stdin=subprocess.PIPE);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24);previews=[]
        try:
            for index in range(count):
                canvas=Image.new('RGB',size,'black');draw=ImageDraw.Draw(canvas)
                for side,(label,directory) in enumerate(zip(spec['labels'],directories)):
                    top=side*(height+36);draw.text((10,top+4),label,font=font,fill='white');step=min(index,counts[side]-1)
                    for column,camera in enumerate(CAMERAS):
                        with Image.open(directory/f'{step:04d}-{camera}.jpg') as frame:
                            assert frame.size==(width,height);canvas.paste(frame,(column*width,top+36))
                child.stdin.write(canvas.tobytes())
                if index in samples:previews.append(canvas.resize((480,276)))
            child.stdin.close();assert child.wait(timeout=30)==0
        except BaseException:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            raise
        subprocess.run([binary,'-v','error','-i',str(video),'-f','null','-'],check=True,timeout=60)
        sheet=Image.new('RGB',(960,828),'white')
        for index,item in enumerate(previews):sheet.paste(item,((index%2)*480,(index//2)*276))
        sheet.save(preview,quality=92);reused=None
    result=dict(key=spec['key'],kind=spec['kind'],comparison=spec['comparison'],seed=spec['seed'],task=spec['task'],scene_seed=spec['scene_seed'],
        labels=spec['labels'],left=spec['left'],right=spec['right'],left_success=rows[0]['success'],right_success=rows[1]['success'],
        video=str(video),video_sha256=sha(video),preview=str(preview),preview_sha256=sha(preview),frames=count,side_frames=counts,preview_indices=samples,
        full_decode_passed=True,reused=reused,seconds=time.monotonic()-started,camera_order=CAMERAS,
        scope='Three actual policy views on each side; action-index alignment with final-frame hold. No external whole-robot camera or exhaustive manual review implied.')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args()
    result=encode(json.loads(Path(args.spec).read_text()));print(json.dumps(dict(key=result['key'],seconds=result['seconds'],reused=bool(result['reused']))))
