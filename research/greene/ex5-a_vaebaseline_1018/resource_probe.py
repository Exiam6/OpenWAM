"""CPU resource amendment: measure allocated memory without driver reservations."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')


def verify():
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    amendment=json.loads((ROOT/'resource-registration-v2.json').read_text())
    assert amendment['gpu_registration_sha256']==digest(ROOT/'gpu-registration.json')
    assert amendment['numerical_stage_files_unchanged'] and amendment['actual_used_memory_limit_mib']==256
    for path,want in amendment['files'].items():assert digest(path)==want,path
    assert str(Path(__file__).resolve()) in amendment['files']
    return amendment


def capacity(p,gpus):
    occupied={line.split(',')[0].strip() for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines() if line.strip()}
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,memory.used,memory.reserved,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    samples={}
    for line in output.splitlines():
        index,uuid,pci,free,used,reserved,util=[x.strip() for x in line.split(',')];gpu=int(index)
        if gpu not in gpus:continue
        assert uuid==p['gpus'][str(gpu)]
        free,used,reserved,util=map(int,[free,used,reserved,util]);assert min(free,used,reserved,util)>=0
        samples[gpu]=dict(gpu=gpu,uuid=uuid,pci=pci,free_mib=free,used_mib=used,driver_reserved_mib=reserved,util=util,
            idle=uuid not in occupied and util==0 and used<256)
    assert set(samples)==set(gpus)
    return samples


def idle_gpus(p):return list(capacity(p,[0,5,6,7]).values())
