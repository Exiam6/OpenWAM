"""Assemble policy-input videos (head | left | right camera, one frame per policy call) from an
evaluation episode's frames/ folder. Pure rendering of logged frames; nothing is re-simulated.
Usage: make_videos.py <episode_dir> <out.mp4>   (needs an ffmpeg binary in FFMPEG env var)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

CAMS = ["head_camera", "left_camera", "right_camera"]
LABELS = ["Head", "Left wrist", "Right wrist"]


def assemble(episode_dir, out_mp4, fps=4):
    frames = Path(episode_dir) / "frames"
    steps = sorted({p.name.split("-")[0] for p in frames.glob("*-head_camera.jpg")})
    if not steps:
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, step in enumerate(steps):
            tiles = []
            for cam, label in zip(CAMS, LABELS):
                f = frames / f"{step}-{cam}.jpg"
                im = Image.open(f).convert("RGB") if f.exists() else Image.new("RGB", (320, 240), (0, 0, 0))
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, 92, 14], fill=(0, 0, 0))
                d.text((3, 1), label, fill=(255, 255, 255))
                tiles.append(im)
            w = sum(t.width for t in tiles)
            h = max(t.height for t in tiles)
            canvas = Image.new("RGB", (w + (w % 2), h + (h % 2)), (0, 0, 0))
            x = 0
            for t in tiles:
                canvas.paste(t, (x, 0))
                x += t.width
            d = ImageDraw.Draw(canvas)
            d.text((w - 70, h - 14), f"step {int(step)}", fill=(255, 255, 255))
            canvas.save(Path(tmp) / f"{i:05d}.jpg", quality=90)
        cmd = [os.environ["FFMPEG"], "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(tmp) / "%05d.jpg"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", str(out_mp4)]
        subprocess.run(cmd, check=True)
    return len(steps)


if __name__ == "__main__":
    print(assemble(sys.argv[1], sys.argv[2]))
