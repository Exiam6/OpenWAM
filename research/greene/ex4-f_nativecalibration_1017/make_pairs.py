"""Encode all completed registered pairs, including losses and unchanged failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
BASE=ROOT.parent/'rae-factorial-20261010'
CAMERAS=['head_camera','left_camera','right_camera']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def make(path):
    result=json.loads(path.read_text())
    if result['status']!='complete':return None
    cell=path.parents[3].name;task=result['task'];seed=result['scene_seed']
    old=BASE/'evaluation/fresh'/cell/task/'clean'/f'seed-{seed}'
    old_result=json.loads((old/'result.json').read_text());assert old_result['status']=='complete'
    assert old_result['initial_state']==result['initial_state']
    dest=ROOT/'paired-videos'/cell/task;dest.mkdir(parents=True,exist_ok=True)
    video=dest/f'seed-{seed}.mp4';metadata=video.with_suffix('.json')
    if metadata.exists():
        saved=json.loads(metadata.read_text());assert saved['new_result_sha256']==sha(path);assert saved['old_result_sha256']==sha(old/'result.json')
        return saved
    assert not video.exists(),'Partial encoding preserved; inspect before regenerating'
    directories=[old/'frames',path.parent/'frames']
    steps=[[int(x.name.split('-')[0]) for x in d.glob('*-head_camera.jpg')] for d in directories]
    for indices,d in zip(steps,directories):
        indices.sort();assert indices==list(range(indices[-1]+1))
        assert all((d/f'{index:04d}-{camera}.jpg').exists() for index in indices for camera in CAMERAS)
    width,height=Image.open(directories[0]/'0000-head_camera.jpg').size
    size=(width*3,(height+36)*2)
    binary=imageio_ffmpeg.get_ffmpeg_exe()
    command=[binary,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','12','-i','-',
        '-an','-c:v','libx264','-threads','2','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    count=max(map(len,steps));preview_steps=sorted(set([0,count//5,2*count//5,3*count//5,4*count//5,count-1]))
    previews=[]
    for step in range(count):
        canvas=Image.new('RGB',size,'black');draw=ImageDraw.Draw(canvas)
        for row,(label,d) in enumerate(zip(['Old','Fixed'],directories)):
            top=row*(height+36);draw.text((10,top+4),label,font=font,fill='white')
            current=min(step,steps[row][-1])
            for col,camera in enumerate(CAMERAS):
                with Image.open(d/f'{current:04d}-{camera}.jpg') as frame:canvas.paste(frame,(col*width,top+36))
        proc.stdin.write(canvas.tobytes())
        if step in preview_steps:
            preview=canvas.resize((480,276));previews.append(preview)
    proc.stdin.close();assert proc.wait()==0
    subprocess.run([binary,'-v','error','-i',str(video),'-f','null','-'],check=True)
    sheet=Image.new('RGB',(960,276*3),'white')
    for i,preview in enumerate(previews):sheet.paste(preview,((i%2)*480,(i//2)*276))
    sheet.save(video.with_suffix('.jpg'),quality=92)
    row=dict(cell=cell,task=task,scene_seed=seed,old_success=old_result['success'],fixed_success=result['success'],
        old_frames=len(steps[0]),fixed_frames=len(steps[1]),frames=count,fps=12,
        alignment='action index, not physical time; shorter episode holds its final frame',
        camera_order=CAMERAS,actual_policy_camera_views=True,external_full_robot_camera=False,
        video=str(video),preview=str(video.with_suffix('.jpg')),preview_indices=preview_steps,
        old_result_sha256=sha(old/'result.json'),new_result_sha256=sha(path),video_sha256=sha(video),full_decode_passed=True)
    metadata.write_text(json.dumps(row,indent=2)+'\n');return row

def main(task,seed):
    records=[]
    for path in sorted((ROOT/'evaluation').glob(f'*-seed{seed}/fixed/{task}/seed-*/result.json')):
        record=make(path)
        if record:records.append(record);print(json.dumps({k:record[k] for k in ['cell','task','scene_seed','old_success','fixed_success','video']}),flush=True)
    (ROOT/f'paired-videos-{task}-seed{seed}.json').write_text(json.dumps(dict(records=records,
        scope='All completed pairs for the specified task and training seed at encode time; pending results omitted explicitly.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();main(a.task,a.seed)
