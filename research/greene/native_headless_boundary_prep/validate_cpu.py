"""Installed-header CPU mocks and compile/link only production helper. No GPU execution."""
import datetime,hashlib,json,os,pathlib,subprocess,sys
S=pathlib.Path(__file__).resolve().parent
H=pathlib.Path('/home/zifanz4');R=H/'openwam-runtime'
I=R/'benchmark-env/lib/python3.10/site-packages/sapien/include'
L=I.parent.parent/'sapien.libs/libsvulkan2.so'
P=R/'ex8-k_glfwprefix_1021/protocol.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 assert not (H/'.local/state/openwam-selfcheck/paused.json').exists()
 assert not any(pathlib.Path(x).exists() for x in json.loads(P.read_text())['pause_paths'])
def main(out):
 guard();assert not out.exists();out.mkdir(parents=True)
 inputs=[S/'PLAN.md',S/'headless.cpp',S/'mock_driver.cpp',S/'validate_cpu.py',L]
 inputs+=list((I/'vulkan').glob('*.h'))+list((I/'vulkan').glob('*.hpp'))
 inputs+=[H/'openwam-native-contract-visibility/research/greene/native_contract_visibility/PREPARATION.json',H/'.local/state/openwam-selfcheck/evidence/20261022T0315Z/cost-after-k.json']
 before={str(p):sha(p) for p in inputs}
 assert before[str(L)]==json.loads(P.read_text())['input_sha256'][str(L)]
 def run(args,env=None):
  guard();r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45,env=env)
  with (out/'commands.log').open('a') as f:f.write(json.dumps(args)+'\n'+r.stdout+'\nexit='+str(r.returncode)+'\n')
  r.check_returncode();return r.stdout
 common=['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(I)]
 run(common+['-DNDEBUG','-shared','-fPIC',str(S/'headless.cpp'),'-L'+str(L.parent),'-l:libsvulkan2.so','-ldl','-o',str(out/'libheadless.so')])
 # Production shared object is NOT loaded. Verify dependency and dispatcher ownership statically.
 prod=run(['readelf','-d',str(out/'libheadless.so')]);assert 'libsvulkan2.so' in prod
 symbols=run(['nm','-D','-C',str(out/'libheadless.so')]);assert 'U vk::defaultDispatchLoaderDynamic' in symbols
 run(common+['-DNDEBUG','-c',str(S/'headless.cpp'),'-o',str(out/'headless.o')])
 run(common+[str(out/'headless.o'),str(S/'mock_driver.cpp'),'-Wl,--wrap=dlopen','-Wl,--wrap=dlsym','-Wl,--wrap=dlclose','-o',str(out/'mock_driver')])
 deps=run(['readelf','-d',str(out/'mock_driver')]);assert not any(x in deps.lower() for x in ['libcuda','libvulkan','libsvulkan','libsapien','libglfw','libglx'])
 undefined=run(['nm','-u',str(out/'mock_driver')]);assert not any(x in undefined for x in [' dlopen',' dlsym',' dlclose'])
 checks=[]
 for case in list(range(13))+list(range(20,28)):
  env=os.environ.copy();env.pop('SAPIEN_VULKAN_LIBRARY_PATH',None)
  if case==5:env['SAPIEN_VULKAN_LIBRARY_PATH']=str(S/'headless.cpp') # regular file; fake dlopen only
  checks.append(run([str(out/'mock_driver'),str(case)],env).strip())
 assert all(sha(pathlib.Path(p))==v for p,v in before.items())
 result=dict(kind='CPU_MOCK_AND_PRODUCTION_COMPILE_LINK_ONLY',updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,checks_passed=len(checks),input_sha256=before,production_sha256=sha(out/'libheadless.so'),mock_sha256=sha(out/'mock_driver'),production_dependencies=prod,mock_dependencies=deps,real_driver_calls=0,GPU_seconds=0,native_L2_L3_passed=False,full_release_allowed=False,full_paper_reproduced=False)
 guard();(out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(checks_passed=len(checks),GPU_seconds=0)))
if __name__=='__main__':main(pathlib.Path(sys.argv[1]))
