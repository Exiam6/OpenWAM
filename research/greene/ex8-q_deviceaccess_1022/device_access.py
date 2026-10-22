"""Additional per-native-thread device-file restrictions; no GPU library imports.
Observer thread must already exist; it retains complete host telemetry access.
This is not an ioctl or hostile-code security boundary on Landlock ABI4.
"""
import ctypes,os,re,stat,threading
from pathlib import Path
from native_adapter import pci
READ=1<<2;WRITE=1<<1;DIR=1<<3;REFER=1<<13;HANDLED=READ|WRITE|DIR|REFER
class Ruleset(ctypes.Structure):_fields_=[('fs',ctypes.c_uint64)]
class Beneath(ctypes.Structure):
 _pack_=1
 _fields_=[('access',ctypes.c_uint64),('fd',ctypes.c_int32)]
def abi():
 lib=ctypes.CDLL(None,use_errno=True);v=lib.syscall(444,0,0,1)
 if v<1:raise OSError(ctypes.get_errno(),'Landlock unavailable')
 return v

def restrict_paths(paths,*,list_dirs=()):
 """Apply only in the calling thread and future children. Irreversible there."""
 assert abi()>=3
 lib=ctypes.CDLL(None,use_errno=True);rules=Ruleset(HANDLED)
 fd=lib.syscall(444,ctypes.byref(rules),ctypes.sizeof(rules),0)
 if fd<0:raise OSError(ctypes.get_errno(),'Create Landlock ruleset')
 try:
  for name,listing_only in [(n,False) for n in sorted(set(map(str,paths)))]+[(str(n),True) for n in list_dirs]:
   pathfd=os.open(name,os.O_PATH|os.O_CLOEXEC)
   try:
    mode=os.fstat(pathfd).st_mode;mask=HANDLED if stat.S_ISDIR(mode) else READ|WRITE
    rule=Beneath(DIR if listing_only else mask,pathfd)
    if lib.syscall(445,fd,1,ctypes.byref(rule),0):raise OSError(ctypes.get_errno(),'Landlock path rule: '+name)
   finally:os.close(pathfd)
  if lib.prctl(38,1,0,0,0):raise OSError(ctypes.get_errno(),'Set no_new_privs')
  if lib.syscall(446,fd,0):raise OSError(ctypes.get_errno(),'Restrict native thread')
 finally:os.close(fd)
 return dict(landlock_abi=abi(),thread_only=True,handled_access_fs=HANDLED)

def assigned_nodes(expected,uuid,*,proc_root=Path('/proc/driver/nvidia/gpus'),drm_root=Path('/sys/class/drm'),dev_root=Path('/dev')):
 target=pci(expected);matches=[]
 for info in proc_root.glob('*/information'):
  data=info.read_text()
  if pci(info.parent.name)==target:
   found=re.search(r'^GPU UUID:\s*(GPU-\S+)\s*$',data,re.M)
   minor=re.search(r'^Device Minor:\s*(\d+)\s*$',data,re.M)
   assert found and found[1]==uuid and minor,'PCI/UUID/minor disagreement'
   matches.append(dev_root/('nvidia'+minor[1]))
 assert len(matches)==1 and matches[0].exists(),'One mapped NVIDIA device required'
 for node in drm_root.iterdir():
  if re.fullmatch(r'(card\d+|renderD\d+)',node.name) and (node/'device').exists():
   try:bus=pci((node/'device').resolve().name)
   except AssertionError:continue
   if bus==target:matches.append(dev_root/'dri'/node.name)
 assert len(matches)==3 and all(p.exists() for p in matches),'Assigned NVIDIA/card/render nodes required'
 return sorted(map(str,matches))

def native_rules(allowed,*,root=Path('/'),dev=Path('/dev')):
 paths=[p for p in root.iterdir() if p.name!='dev']
 # Never grant /dev, /dev/dri, aliases to them or peer NVIDIA character files.
 for p in dev.iterdir():
  if p.is_symlink():continue
  if p.name=='dri' or re.fullmatch(r'nvidia\d+',p.name):continue
  paths.append(p)
 paths.extend(Path(p) for p in allowed)
 return paths

def apply(expected,uuid):
 assert threading.current_thread() is threading.main_thread()
 assert threading.active_count()>=2,'Observer must start before native restriction'
 # No inherited GPU descriptors may bypass the read/write file policy.
 existing=[]
 for name in Path('/proc/self/fd').iterdir():
  try:s=os.fstat(int(name.name))
  except OSError:continue
  if stat.S_ISCHR(s.st_mode) and os.major(s.st_rdev) in (195,226,507):existing.append(str(name))
 assert not existing,'GPU descriptors exist before restriction'
 allowed=assigned_nodes(expected,uuid)
 proof=restrict_paths(native_rules(allowed),list_dirs=[Path("/dev")])
 return dict(proof,allowed_device_nodes=allowed,expected_pci=expected,assigned_uuid=uuid,
             limitation='ABI4 does not restrict ioctl on inherited/global control descriptors; not a security proof; all-card observation still mandatory')
