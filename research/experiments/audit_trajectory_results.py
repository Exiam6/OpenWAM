"""Read-only post-run audit: bytes, decoded camera inputs and native action conversion."""
import base64,datetime,gzip,hashlib,io,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/data02/zifanz4/openwam-experiments');RUN=ROOT/'results/repeatability-20260921/closedloop';PARENT=RUN.parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'OpenWAM')]
from repeat_common import array_hash,payload_hash
from benchmarks.utils.action_conversion import eef20d_to_ee16d
s=json.loads((RUN/'summary.json').read_text());freeze=json.loads((RUN/'source-freeze.json').read_text());assert s['all_integrity_checks_passed'] and s['episodes']==12
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert(RUN/'exit-code.txt').read_text().strip()=='0'
assert all(sha(ROOT/k)==v for k,v in freeze['sources'].items())
for k,name in [('capture_sha256','capture-summary.json'),('parent_summary_sha256','summary.json'),('manifest_sha256','scenes.json')]:assert sha(PARENT/name)==freeze[k]
assert (RUN/'round0-effective-config.yaml').read_bytes()==(RUN/'round1-effective-config.yaml').read_bytes()
checks=json.loads((RUN/'cpu-checks.json').read_text());assert len(checks)==9 and all(checks.values())
manifest=json.loads((PARENT/'scenes.json').read_text());scenes={x['seed']:x for x in manifest['scenes']}
camera_map={'head_camera':'head_camera','left_camera':'left_wrist_camera','right_camera':'right_wrist_camera'}
total=0;videos=[];all_pairs=[]
for group,data in s['groups'].items():
 d=RUN/group;prov=json.loads((d/'provenance.json').read_text());assert prov['arm']=='identity'and prov['role']=='trajectory_diagnostic'and prov['manifest_sha256']==freeze['manifest_sha256']
 assert prov['noise_sigma']==(0.2 if group.endswith('noise')else 0)
 for ep in data['records']:
  assert ep['instruction']==scenes[ep['seed']]['instruction']
  records=[json.loads(x)for x in(d/f"seed-{ep['seed']}-trace.jsonl").read_text().splitlines()]
  for rec in records:
   payload=json.loads(gzip.decompress((d/rec['request_file']).read_bytes()))
   assert payload_hash(payload)==rec['request_sha256']
   for camera,key in camera_map.items():
    arr=np.asarray(Image.open(io.BytesIO(base64.b64decode(payload['images'][key]))).convert('RGB'))
    assert array_hash(arr)==rec['policy_cameras'][camera]
   state=np.asarray(payload['state'],dtype=np.float32);assert array_hash(state)==rec['eef_hash']
   converted=eef20d_to_ee16d(np.asarray(rec['server_action'],dtype=np.float32))
   assert array_hash(converted)==rec['env_action_hash']
   total+=1
 for v in sorted((d/'runtime/eval_result').rglob('episode*.mp4')):
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,nb_frames','-of','json',str(v)],text=True));assert float(probe['format']['duration'])>0
  stream=probe['streams'][0];assert int(stream['nb_frames'])>0 and stream['width']>0 and stream['height']>0
  videos.append({'path':str(v.relative_to(RUN)),'sha256':sha(v),'duration':float(probe['format']['duration']),'frames':int(stream['nb_frames'])})
assert total==s['requests']and len(videos)==12
out={'audited_at':datetime.datetime.now().astimezone().isoformat(),'all_checks_passed':True,'unchanged_frozen_sources':len(freeze['sources']),'parent_capture_summary_and_manifest_unchanged':True,'two_server_configs_byte_identical':True,'fresh_cpu_checks':9,'requests':total,'decoded_camera_images':total*3,'native_action_conversions_exact':total,'proprio_payloads_exact':total,'videos':videos,'script_sha256':sha(Path(__file__))}
(RUN/'supplemental-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='videos'},indent=2))
