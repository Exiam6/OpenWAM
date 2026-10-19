"""Read-only inventory. No CUDA context, changes to compute mode, or process signals."""
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess


def query(fields, kind='gpu'):
    result = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=8, check=True)
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(result.stdout))]


def probe():
    applications = []
    for uuid, pid, memory in query('gpu_uuid,pid,used_memory', 'compute-apps'):
        item = dict(uuid=uuid, pid=int(pid), memory=memory, user='unknown', pgid=None)
        try:
            item.update(user=pwd.getpwuid((Path('/proc')/pid).stat().st_uid).pw_name,
                        pgid=os.getpgid(int(pid)))
        except (FileNotFoundError, ProcessLookupError, PermissionError, KeyError):
            pass
        applications.append(item)
    rows = []
    for index, uuid, util, used, total in query('index,uuid,utilization.gpu,memory.used,memory.total'):
        apps = [a for a in applications if a['uuid']==uuid]
        rows.append(dict(index=int(index), uuid=uuid, utilization=int(util), used_mib=int(used),
                         total_mib=int(total), processes=apps,
                         idle=not apps and int(util)==0 and int(used)<256))
    assert len(rows)==8 and {r['index'] for r in rows}==set(range(8)), 'Expected all eight GPUs'
    return dict(hostname=socket.getfqdn(), time=dt.datetime.now(dt.timezone.utc).isoformat(), gpus=rows)


if __name__=='__main__':
    print(json.dumps(probe()))
