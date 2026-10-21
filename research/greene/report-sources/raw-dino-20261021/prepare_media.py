import json,pathlib,concurrent.futures,subprocess,hashlib,shutil,time
ROOT=pathlib.Path('/home/zifanz4'); R=ROOT/'openwam-runtime'; V=R/'ex5-a_vaebaseline_1018/full-report'; OUT=pathlib.Path(__file__).parent/'dist'; FF=R/'policy-env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
rows=json.loads((V/'index.json').read_text())
def one(it):
 i,row=it; meta=json.loads((V/row['metadata']).read_text()); src=V/row['video']; out=OUT/'media'/f'{i:03d}.mp4'; preview=OUT/'media'/f'{i:03d}.jpg'
 assert hashlib.sha256(src.read_bytes()).hexdigest()==meta['video_sha256']
 if not out.exists():
  tmp=out.with_suffix('.tmp.mp4');subprocess.run([str(FF),'-v','error','-y','-i',str(src),'-vf','scale=640:-2','-c:v','libx264','-preset','fast','-crf','34','-threads','2','-an','-movflags','+faststart',str(tmp)],check=True);tmp.rename(out)
 subprocess.run([str(FF),'-v','error','-i',str(out),'-f','null','-'],check=True)
 if not preview.exists():subprocess.run([str(FF),'-v','error','-i',str(out),'-frames:v','1','-q:v','5',str(preview)],check=True)
 dest=OUT/'data'/f'pair-{i:03d}.json';shutil.copyfile(V/row['metadata'],dest)
 return dict(row,video=f'media/{out.name}',preview=f'media/{preview.name}',metadata=f'data/{dest.name}',original_video_sha256=meta['video_sha256'],web_video_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),frames=meta['frames'])
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 result=[]
 for row in pool.map(one,enumerate(rows)):
  result.append(row)
  if len(result)%30==0: print('verified',len(result),flush=True)
(OUT/'data/clips.json').write_text(json.dumps(result,indent=2));print('COMPLETE',len(result),sum(p.stat().st_size for p in (OUT/'media').glob('*')),flush=True)
