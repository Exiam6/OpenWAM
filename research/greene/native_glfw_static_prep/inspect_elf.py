"""CPU-only installed-ELF evidence. Never imports or loads a native library."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

LIB = Path('/home/zifanz4/openwam-runtime/benchmark-env/lib/python3.10/site-packages/sapien.libs/libsvulkan2.so')
PROTOCOL = Path('/home/zifanz4/openwam-runtime/ex8-k_glfwprefix_1021/protocol.json')
ROOT = Path('/home/zifanz4/.local/state/openwam-selfcheck/evidence')
SYMBOLS = ['glfwInit', 'glfwVulkanSupported', '_glfwInitVulkan', '_glfwPlatformInit', '_glfwInitGLX']

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def unpaused():
    assert not Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists(), 'Global pause'
    assert not any(Path(x).exists() for x in json.loads(PROTOCOL.read_text())['pause_paths']), 'Inherited pause'

def symbol(table, name):
    found=[]
    for line in table.splitlines():
        fields=line.split()
        if len(fields)==7 and fields[1:4]==['g','DF','.text'] and fields[-1]==name:
            found.append((int(fields[0],16),int(fields[4],16)))
    assert len(found)==1 and found[0][1]>0, 'Missing or ambiguous function: '+name
    return found[0]

def literal(raw, address):
    assert raw[:6]==b'\x7fELF\x02\x01', 'Requires ELF64 little endian'
    offset=struct.unpack_from('<Q',raw,40)[0]
    size,count=struct.unpack_from('<HH',raw,58)
    assert size==64
    for index in range(count):
        section=struct.unpack_from('<IIQQQQIIQQ',raw,offset+index*size)
        kind,base,start,length=section[1],section[3],section[4],section[5]
        if kind!=8 and base<=address<base+length:
            loc=start+address-base
            end=raw.find(b'\0',loc,min(start+length,loc+512))
            if end<0:return None
            value=raw[loc:end]
            return value.decode('ascii') if value and all(32<=x<127 for x in value) else None
    return None

def tail_transfers(text):
    return [line.strip() for line in text.splitlines() if re.search(r'\bjmp[q]?\s',line) and '@plt>' in line]

def destination(path):
    path=path.resolve()
    assert path.is_relative_to(ROOT.resolve()) and path!=ROOT.resolve(), 'Evidence output only'
    return path

def write(path, value):
    unpaused()
    path=destination(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    unpaused()
    with path.open('x') as f:f.write(value)

def inspect(output):
    output=destination(output)
    raw=LIB.read_bytes()
    protocol=json.loads(PROTOCOL.read_text())
    assert protocol['input_sha256'][str(LIB)]==sha(raw), 'Frozen installed binary mismatch'
    def run(*args):return subprocess.check_output(args,text=True)
    table=run('objdump','-T',str(LIB))
    result=dict(kind='CPU_STATIC_EVIDENCE_NOT_GPU_VALIDATION',updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),library=str(LIB),library_sha256=sha(raw),build_id=run('readelf','-n',str(LIB)),source_sha256=sha(Path(__file__).read_bytes()),functions={},native_libraries_loaded=False,full_release_allowed=False,full_paper_reproduced=False)
    for name in SYMBOLS:
        start,length=symbol(table,name)
        text=run('objdump','-d','-C','--start-address='+hex(start),'--stop-address='+hex(start+length),str(LIB))
        calls=[line.strip() for line in text.splitlines() if re.search(r'\bcall[q]?\s',line)]
        refs=[]
        for line in text.splitlines():
            match=re.search(r'# ([0-9a-f]+) ',line)
            if match:
                addr=int(match[1],16);value=literal(raw,addr)
                if value:refs.append(dict(address=hex(addr),value=value,instruction=line.strip()))
        result['functions'][name]=dict(address=hex(start),size=length,calls=calls,tail_transfers=tail_transfers(text),referenced_literals=refs,disassembly_sha256=sha(text.encode()))
        write(output/(name+'.asm'),text)
    assert sha(LIB.read_bytes())==result['library_sha256'], 'Binary changed during inspection'
    result['limitations']='Static direct instructions only; not an executed trace. Indirect and transitive execution, exact driver causality, runtime loader resolution and matching upstream binary source revision remain unproven.'
    write(output/'call-path.json',json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=inspect(a.output)
    print(json.dumps({name:dict(address=v['address'],calls=len(v['calls']),literals=[x['value'] for x in v['referenced_literals']]) for name,v in r['functions'].items()},indent=2))
