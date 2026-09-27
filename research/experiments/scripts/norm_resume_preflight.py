"""Narrow allowance for the identified same-user G1 graphics-only context."""
import json,os,socket,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/norm-20260921'
UUID='GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3'
assert os.environ['CUDA_VISIBLE_DEVICES']==UUID
samples=[];valid=0
for attempt in range(24):
    gpu=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x','-i',UUID],text=True)).find('gpu')
    assert gpu.findtext('uuid')==UUID
    memory,util,ecc=map(int,subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).split(','))
    processes=[];allowed=True
    for info in gpu.findall('./processes/process_info'):
        pid=int(info.findtext('pid'));kind=info.findtext('type');used=int(info.findtext('used_memory').split()[0])
        try:
            own=os.stat(f'/proc/{pid}').st_uid==os.getuid()
            argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            recognized=(b'/data02/zifanz4/g1-stage2-runtime-20260921/SIMPLE/.venv/bin/python' in argv and b'/data02/zifanz4/g1-stage2-runtime-20260921/uneven-support-controlled-20260921/terrain_eval_entry.py' in argv)
        except FileNotFoundError:
            own=recognized=False
        ok=kind=='G' and used<=384 and own and recognized
        allowed=allowed and ok
        processes.append({'pid':pid,'type':kind,'memory_mib':used,'owned':own,'recognized_graphics_context':recognized,'allowed':ok})
    sample={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'memory_mib':memory,'utilization':util,'ecc':ecc,'processes':processes}
    samples.append(sample)
    (OUT/'resume-preflight.json').write_text(json.dumps(samples,indent=2))
    ok=allowed and len(processes)<=1 and memory<=512 and util<=15 and ecc==0
    valid=valid+1 if ok else 0
    if valid>=3:break
    time.sleep(5)
else:raise RuntimeError('Same-GPU resumed preflight did not pass; samples saved')
s=socket.socket();s.bind(('127.0.0.1',18848));s.close()
print('RESUME_PREFLIGHT_PASSED',json.dumps(samples[-1]),flush=True)
