"""CPU import-safe headless composition; no executable GPU entrypoint."""
from pathlib import Path
import hashlib
import json
import sys
from supervisor import Supervisor, OPERATIONS

DEPENDENCIES=Path('/home/zifanz4/openwam-native-contract-visibility/research/greene')
sys.path.insert(0,str(DEPENDENCIES/'native_evidence_adapter'))
from adapter import DurableObserver, Journal

HELPER=(('native_glfwInit','headless_init',None),
        ('native_headless_support','headless_support',0),
        ('native_loader','headless_loader',1),('native_getproc','headless_getproc',1),
        ('native_dispatcher','headless_dispatcher',1),
        ('native_layer_symbol','headless_layer_symbol',1),
        ('native_layer_count','headless_layer_count',1),
        ('native_layer_data','headless_layer_data',1))

def operations(backend):
    # Backend must retain native objects for the entire sequence. No library is
    # loaded here until the supervised helper_load callback actually executes.
    out={name:getattr(backend,name) for name in OPERATIONS[:4]}
    def call(symbol,expected):
        value=backend.helper_call(symbol)
        if expected is None:
            if value not in (0,1):raise RuntimeError('Invalid glfwInit result')
            return True
        if value!=expected:raise RuntimeError('Native boundary failed: '+symbol)
        return True
    out.update({name:lambda symbol=symbol,expected=expected:call(symbol,expected)
                for name,symbol,expected in HELPER})
    assert tuple(out)==OPERATIONS
    return out

def run(backend,observe,guard,assigned,*,seconds=70,sample_timeout=14,interval=.02):
    return Supervisor().run(operations(backend),observe,guard,assigned,
          seconds=seconds,sample_timeout=sample_timeout,interval=interval)

def observer(config,argv,journal,*,seconds=4):
    return DurableObserver(argv,Journal(journal),host=config['host'],
            device_map={int(k):v for k,v in config['device_map'].items()},
            assigned=config['assigned'],own_group=config['own_group'],
            gdm_uid=config['gdm_uid'],seconds=seconds)

def outcome(scientific,worker,claim):
    out=dict(GPU_diagnostic_validated=False,native_L2_L3_passed=False,
             full_release_allowed=False,full_paper_reproduced=False)
    if not all([scientific,worker,claim]):return dict(out,assessment='pending evidence')
    if (worker.get('id')!=claim.get('id') or not worker.get('scientific_identity')
        or worker.get('scientific_identity')!=claim.get('scientific_identity')
        or not claim.get('gpu_uuid') or worker.get('gpu_uuid')!=claim.get('gpu_uuid')):
        return dict(out,assessment='identity mismatch')
    if (scientific.get('status')!='supervisor_sequence_passed'
        or worker.get('status')!='complete' or claim.get('status')!='complete'
        or type(worker.get('exit_code')) is not int or worker['exit_code']!=0
        or scientific.get('monitor_errors')):
        return dict(out,assessment='failed or inconsistent terminal evidence')
    assigned=claim['gpu_uuid'];samples=scientific.get('samples',[])
    required=['before_prewarm']+['after_'+n for n in OPERATIONS]
    counts={phase:sum(s.get('phase')==phase and s.get('finished_phase')==phase
                      and s.get('stable_phase') is True and s.get('complete') is True
                      for s in samples) for phase in required}
    # Raw packets remain in the journal: never substitute summaries for them.
    receipts_ok=True
    for sample in samples:
        receipt=sample.get('evidence_receipt',{})
        try:
            raw=Path(receipt['path']).read_bytes()
            record=json.loads(raw)
            assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
            assert record['kind']=='validated_sample'
            assert record['packet']['evidence']['cleanup_reaped'] is True
        except Exception:receipts_ok=False
    contexts=[c for sample in samples for c in sample.get('contexts',[])]
    passed=(all(n>=3 for n in counts.values()) and receipts_ok
            and bool(contexts) and all(c.get('uuid')==assigned for c in contexts)
            and scientific.get('results')=={n:True for n in OPERATIONS})
    return dict(out,GPU_diagnostic_validated=passed,phase_counts=counts,
                assessment='headless prefix only; native scene validation required' if passed
                else 'incomplete scientific evidence')

def timing(cap,*,sample_seconds,phase_count=None,native_seconds=0,setup_seconds=0):
    # Frozen dispatcher: admission elapsed < cap-60; child loop elapsed <cap-45.
    phases=len(OPERATIONS)+1 if phase_count is None else phase_count
    measured_estimate=setup_seconds+native_seconds+3*phases*sample_seconds+45
    return dict(cap_seconds=cap,admission_latest_seconds=cap-60,
                child_latest_elapsed_seconds=cap-45,stable_samples_required=3*phases,
                projected_occupied_seconds=measured_estimate,
                projected_to_fit=measured_estimate<cap,
                timing_is_estimate_not_guarantee=True)

class NativeBackend:
    """Lazy native operations; caller must enforce frozen registration/dispatcher."""
    def __init__(self,helper_path,expected_pci):
        self.helper_path=helper_path;self.expected_pci=expected_pci;self.details={}
    def prewarm(self):
        import importlib.util
        p=Path('/home/zifanz4/openwam-runtime/ex8-a_officialrepro_1021/official/benchmarks/robotwin/eval_policy_wrapper.py')
        spec=importlib.util.spec_from_file_location('native_wrapper',p)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from prewarm import checked_prewarm
        self.details['prewarm_stdout']=checked_prewarm(module._prewarm_cuda_for_curobo)
        return True
    def sapien_import(self):
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        self.tensor=torch.zeros(1,device='cuda')
        import sapien
        self.sapien=sapien
        return True
    def explicit_device(self):
        self.device=self.sapien.Device('cuda:0')
        from native_adapter import verify_device
        self.details['device']=verify_device(self.device,self.expected_pci)
        return True
    def helper_load(self):
        import ctypes
        self.lib=ctypes.CDLL(str(self.helper_path))
        self.lib.headless_error.restype=ctypes.c_char_p
        for _,symbol,_ in HELPER:getattr(self.lib,symbol).restype=ctypes.c_int
        return True
    def helper_call(self,symbol):
        value=getattr(self.lib,symbol)()
        self.details[symbol]=value
        if value<0:self.details['native_error']=self.lib.headless_error().decode(errors='replace')
        return value
