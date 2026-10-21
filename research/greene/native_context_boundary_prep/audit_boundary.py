"""Read-only CPU audit; never runs/imports a GPU library or releases a queue."""
import collections,hashlib,json,math,pathlib
R=pathlib.Path('/home/zifanz4/openwam-runtime')
def read(p):return json.loads(p.read_text())
def audit():
 native=read(R/'ex8-h_deviceboundary_1021/result.json')
 worker=read(R/'nightly-gpu/runs/ex8h-native-device-boundary/result.json')
 claim=read(R/'nightly-gpu/claims/ex8h-native-device-boundary.json')
 assert worker['status']==claim['status']=='failed' and worker['finished'] and worker['exit_code']==1
 assigned=worker['gpu_uuid']; samples=native['samples']
 device=[s for s in samples if s['phase']=='after_explicit_Device']
 assert len(device)>=3 and all(all(c['uuid']==assigned for c in s['contexts']) for s in device)
 peer=next(s for s in samples if any(c['uuid']!=assigned for c in s['contexts']))
 assert peer['phase']=='explicit_RenderSystem'
 assert native['discovered_device']==native['explicit_device']
 assert 'prewarmed CUDA/Curobo before SAPIEN import' in native['prewarm_stdout']
 assert native['status']=='failed' and native['monitor_errors']
 terminal=[]
 for f in (R/'nightly-gpu/runs').glob('ex8*/result.json'):
  if f.parent.name.startswith(('ex8d-','ex8e-','ex8f-','ex8g-','ex8h-')):
   v=read(f);assert v['status'] in ['complete','failed'] and v['finished'];terminal.append(v)
 held=[]
 for entry in read(R/'nightly-gpu/held-for-render-validation-20261021/manifest.json')['entries']:
  f=pathlib.Path(entry['held']);assert hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256'];assert not pathlib.Path(entry['original']).exists();held.append(read(f))
 assert len(held)==97
 used=sum(math.ceil(v['occupied_parent_seconds']) for v in terminal);caps=sum(v['max_seconds'] for v in held)
 total=used+caps+18749+9000;assert total<=280*3600
 return dict(scientific_status=native['status'],worker_status=worker['status'],claim_status=claim['status'],first_peer_sample=peer,phase_samples=dict(collections.Counter(s['phase'] for s in samples)),device_only_samples=len(device),prewarm_succeeded=True,observed_boundary='RenderSystem construction after successful Device boundary',exact_internal_cause='unresolved',cost=dict(terminal_rounded_seconds=used,held_caps_seconds=caps,conditional_blocks_recovery_seconds=18749,conditional_native_validation_seconds=9000,total_reserved_seconds=total,total_reserved_gpu_hours=total/3600,remaining_unallocated_seconds=280*3600-total),full_release_allowed=False,full_paper_reproduced=False)
if __name__=='__main__':print(json.dumps(audit(),indent=2))
