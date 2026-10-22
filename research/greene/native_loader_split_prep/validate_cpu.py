"""Build actual helper with link-time fake loader calls; no real native load."""
import datetime,hashlib,json,pathlib,subprocess,sys
S=pathlib.Path(__file__).resolve().parent
H=pathlib.Path('/home/zifanz4')
P=H/'openwam-runtime/ex8-k_glfwprefix_1021/protocol.json'
HEADER=H/'openwam-runtime/benchmark-env/lib/python3.10/site-packages/sapien/include/vulkan/vulkan_core.h'
LIB=H/'openwam-runtime/benchmark-env/lib/python3.10/site-packages/sapien.libs/libsvulkan2.so'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 assert not (H/'.local/state/openwam-selfcheck/paused.json').exists(),'Global pause'
 assert not any(pathlib.Path(x).exists() for x in json.loads(P.read_text())['pause_paths']),'Inherited pause'
def main(output):
 output=output.resolve();assert output.is_relative_to(H/'.local/state/openwam-selfcheck/evidence')
 assert not output.exists(),'Preserve previous validation'
 assert sha(LIB)==json.loads(P.read_text())['input_sha256'][str(LIB)],'Native ELF changed'
 before={str(p):sha(p) for p in [HEADER,HEADER.with_name('vk_platform.h'),S/'loader_split.cpp',S/'mock_driver.cpp',S/'validate_cpu.py']}
 guard();output.mkdir(parents=True)
 def run(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=45)
 common=['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(HEADER.parent.parent)]
 guard();run(common+['-c',str(S/'loader_split.cpp'),'-o',str(output/'loader_split.o')])
 guard();run(common+[str(S/'loader_split.cpp'),str(S/'mock_driver.cpp'),'-Wl,--wrap=dlopen','-Wl,--wrap=dlsym','-o',str(output/'mock_driver')])
 deps=run(['readelf','-d',str(output/'mock_driver')])
 assert not any(name in deps.lower() for name in ['libcuda','libvulkan','libsvulkan','libsapien','libglfw','libglx'])
 undefined=run(['nm','-u',str(output/'mock_driver')]);assert not any(x in undefined for x in [' dlopen',' dlsym']), 'Real loader reference'
 checks=[]
 for scenario in list(range(9))+list(range(10,15)):
  guard();checks.append(run([str(output/'mock_driver'),str(scenario)]).strip())
 assert all(sha(pathlib.Path(p))==v for p,v in before.items()),'Inputs changed'
 result=dict(kind='CPU_MOCK_VALIDATION_ONLY',updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,checks_passed=len(checks),input_sha256=before,native_elf_sha256=sha(LIB),compiler=run(['g++','--version']),dependencies=deps,undefined_symbols=undefined,object_sha256=sha(output/'loader_split.o'),mock_executable_sha256=sha(output/'mock_driver'),real_GPU_driver_calls=0,GPU_execution_validated=False,registered_GPU_attempt=False,native_L2_L3_passed=False,full_release_allowed=False,full_paper_reproduced=False)
 guard();(output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['kind','checks_passed','real_GPU_driver_calls','GPU_execution_validated']}))
if __name__=='__main__':main(pathlib.Path(sys.argv[1]))
