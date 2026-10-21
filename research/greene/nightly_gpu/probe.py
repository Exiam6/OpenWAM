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
import xml.etree.ElementTree as ET


def query(fields, kind='gpu'):
    result = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=8, check=True)
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(result.stdout))]


def system_display(item):
    return (item['type']=='G' and item['user'] in ['root','gdm']
            and item['name']=='/usr/lib/xorg/Xorg' and 0<=item['memory_mib']<=16)


def probe():
    applications = []
    xml=subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,timeout=8,check=True)
    nodes=ET.fromstring(xml.stdout).findall('gpu')
    assert len(nodes)==8
    for gpu in nodes:
        uuid=gpu.findtext('uuid');assert gpu.find('processes') is not None
        for process in gpu.findall('./processes/process_info'):
            pid=process.findtext('pid');memory=process.findtext('used_memory')
            item=dict(uuid=uuid,pid=int(pid),memory=memory,memory_mib=int(memory.split()[0]),
                      type=process.findtext('type'),name=process.findtext('process_name'),user='unknown',pgid=None)
            try:
                item.update(user=pwd.getpwuid((Path('/proc')/pid).stat().st_uid).pw_name,
                            pgid=os.getpgid(int(pid)))
            except (FileNotFoundError,ProcessLookupError,PermissionError,KeyError):pass
            applications.append(item)
    rows = []
    for index, uuid, util, used, total in query('index,uuid,utilization.gpu,memory.used,memory.total'):
        apps = [a for a in applications if a['uuid']==uuid and not system_display(a)]
        background = [a for a in applications if a['uuid']==uuid and system_display(a)]
        rows.append(dict(index=int(index), uuid=uuid, utilization=int(util), used_mib=int(used),
                         total_mib=int(total), processes=apps,system_display_processes=background,
                         idle=not apps and int(util)==0 and int(used)<256))
    assert len(rows)==8 and {r['index'] for r in rows}==set(range(8)), 'Expected all eight GPUs'
    return dict(hostname=socket.getfqdn(), time=dt.datetime.now(dt.timezone.utc).isoformat(), gpus=rows)


if __name__=='__main__':
    print(json.dumps(probe()))
