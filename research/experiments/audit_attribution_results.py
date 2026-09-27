"""Independent read-only request/conversion/media audit, no simulator or GPU use."""
import base64,datetime,gzip,hashlib,io,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/data02/zifanz4/openwam-experiments');RUN=ROOT/'results/attribution-20260921'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'OpenWAM')]
from repeat_common import array_hash,payload_hash
from benchmarks.utils.action_conversion import eef20d_to_ee16d
s=json.loads((RUN/'summary.json').read_text());freeze=json.loads((RUN/'source-freeze.json').read_text());assert s['all_integrity_checks_passed']and s['commands_executed']==12
assert(RUN/'exit-code.txt').read_text().strip()=='0';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/k)==v for k,v in freeze['sources'].items())
checks=json.loads((RUN/'cpu-checks.json').read_text());assert len(checks)==9 and all(checks.values())
count=0;plans=0;videos=[];moved=0
for rid in[0,1]:
 d=RUN/f'round{rid}'
 for seed in[400000,400001,400002]:
  rows=[json.loads(x)for x in(d/f'seed-{seed}-trace.jsonl').read_text().splitlines()]
  for row in rows:
   p=json.loads(gzip.decompress((d/row['request_file']).read_bytes()));assert payload_hash(p)==row['request_sha256']
   for camera,key in [('head_camera','head_camera'),('left_camera','left_wrist_camera'),('right_camera','right_wrist_camera')]:
    a=np.asarray(Image.open(io.BytesIO(base64.b64decode(p['images'][key]))).convert('RGB'));assert array_hash(a)==row['policy_cameras'][camera]
   assert array_hash(np.asarray(p['state'],np.float32))==row['eef_hash']
   assert array_hash(eef20d_to_ee16d(np.asarray(row['server_action'],np.float32)))==row['env_action_hash']
   # Probe records physical evolution; equal repeats must not be confused with no action.
   before=row['before_execution']['actual_articulations']['left']['qpos'];after=row['after_execution']['actual_articulations']['left']['qpos']
   moved+=before['sha256']!=after['sha256'];plans+=len(row['planner_calls']);count+=1
 for video in(d/'runtime/eval_result').rglob('episode*.mp4'):
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=nb_frames,width,height','-of','json',str(video)],text=True));assert int(info['streams'][0]['nb_frames'])==2 and float(info['format']['duration'])>0
  videos.append({'path':str(video.relative_to(RUN)),'sha256':sha(video),'frames':2,'duration':float(info['format']['duration'])})
assert count==12 and plans==24 and len(videos)==6
out={'audited_at':datetime.datetime.now().astimezone().isoformat(),'all_checks_passed':True,'frozen_source_files':len(freeze['sources']),'requests_and_native_conversions':count,'decoded_camera_images':count*3,'planner_calls':plans,'commands_with_measured_qpos_change':moved,'videos':videos,'script_sha256':sha(Path(__file__))}
(RUN/'supplemental-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='videos'},indent=2))
