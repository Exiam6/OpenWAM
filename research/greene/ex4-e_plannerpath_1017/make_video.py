"""Full paired replay by action index; final frame held for the shorter replay."""
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/zifanz4/openwam-runtime/ex4-e_plannerpath_1017')
DEMO = 'handover_block-seed4101100'
OUTPUT = ROOT / f'paired-{DEMO}.mp4'
frames = {c: sorted((ROOT / 'replay' / c / DEMO / 'frames' / DEMO).glob('*.jpg')) for c in ('old', 'fixed')}
assert all(frames.values())
width, height = Image.open(frames['old'][0]).size
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
binary = imageio_ffmpeg.get_ffmpeg_exe()
command = [binary, '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{2*(height+36)}', '-r', '12', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '4', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUTPUT)]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
total = max(map(len, frames.values()))
for index in range(total):
    canvas = Image.new('RGB', (width, 2*(height+36)), 'black')
    draw = ImageDraw.Draw(canvas)
    for row, condition in enumerate(('old', 'fixed')):
        top = row*(height+36)
        frame = Image.open(frames[condition][min(index, len(frames[condition])-1)])
        canvas.paste(frame, (0, top+36))
        draw.text((10, top+4), condition.title(), fill='white', font=font)
    proc.stdin.write(canvas.tobytes())
    if index in (0, total//4, total//2, 3*total//4, total-1):
        canvas.save(ROOT / f'paired-preview-{index:04d}.jpg')
proc.stdin.close()
assert proc.wait() == 0
subprocess.run([binary, '-v', 'error', '-i', str(OUTPUT), '-f', 'null', '-'], check=True)
(ROOT / 'paired-video.json').write_text(json.dumps(dict(path=str(OUTPUT), frames=total, fps=12,
    source_frames={k:len(v) for k,v in frames.items()}, alignment='expert action index, not physical time; shorter replay holds final frame',
    full_decode_passed=True, conditions='identical YAML bytes, old path vs restored native calibration branch'), indent=2))
print(OUTPUT)
