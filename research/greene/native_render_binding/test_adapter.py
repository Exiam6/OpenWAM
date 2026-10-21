import copy
import types
import unittest
from adapter import delegate, guard, pci

UUID="GPU-89770bc8-d310-99d8-b3a4-d2b14c93d1a9"
ENV={"CUDA_VISIBLE_DEVICES":UUID,"OPENWAM_GPU_HOST":"gpu01","OPENWAM_ALLOCATION_END":"2026-10-21T23:00:00+00:00"}
ARGS=["--config","native.yml"]
class Tests(unittest.TestCase):
    def harness(self, mutate=None, raises=False):
        events=[]
        device=types.SimpleNamespace(cuda_id=0,pci_string="0000:41:00.0",name="assigned",is_cuda=lambda:True,can_render=lambda:True,is_cpu=lambda:False)
        if mutate:mutate(device)
        wrapper=types.SimpleNamespace(_prewarm_cuda_for_curobo=lambda:events.append("prewarm"))
        original=wrapper._prewarm_cuda_for_curobo
        def native(argv):
            self.assertEqual(argv,ARGS);events.append("native")
            wrapper._prewarm_cuda_for_curobo();events.append("scenes")
            if raises:raise RuntimeError("native failure")
            return 17
        wrapper.main=native
        def factory(**kwargs):
            self.assertEqual(kwargs,{"device":"cuda:0"});events.append("render_import_and_create")
            return types.SimpleNamespace(device=device)
        return wrapper,original,factory,events
    def test_native_order_and_restoration(self):
        w,orig,f,events=self.harness()
        self.assertEqual(delegate(w,ARGS,ENV,"00000000:41:00.0",f,lambda _:events.append("record")),17)
        self.assertEqual(events,["native","prewarm","render_import_and_create","record","scenes"])
        self.assertIs(w._prewarm_cuda_for_curobo,orig)
    def test_contaminated_overrides(self):
        from adapter import BLOCKED
        for key in BLOCKED|{"ROBOTWIN_LABTASKER_OPERATION","ROBOTWIN_LABTASKER_RESULT"}:
            with self.subTest(key=key),self.assertRaises(AssertionError):guard(dict(ENV,**{key:"0"}),ARGS)
    def test_bad_gpu_assignments(self):
        for value in ["", "0", UUID+",GPU-other", "GPU-nope"]:
            with self.subTest(value=value),self.assertRaises(AssertionError):guard(dict(ENV,CUDA_VISIBLE_DEVICES=value),ARGS)
    def test_missing_dispatch_allocation(self):
        for key in ["OPENWAM_GPU_HOST","OPENWAM_ALLOCATION_END"]:
            e=copy.copy(ENV);e.pop(key)
            with self.assertRaises(AssertionError):guard(e,ARGS)
    def test_alternate_argv(self):
        for args in [[],["labtasker"],ARGS+["--num-episodes","1"]]:
            with self.assertRaises(AssertionError):guard(ENV,args)
    def test_wrong_device_fail_closed_and_restore(self):
        for field,value in [("cuda_id",1),("pci_string","0000:42:00.0"),("pci_string",None),("is_cuda",lambda:False),("can_render",lambda:False),("is_cpu",lambda:True)]:
            w,orig,f,events=self.harness(lambda d:setattr(d,field,value))
            with self.subTest(field=field),self.assertRaises(AssertionError):delegate(w,ARGS,ENV,"0000:41:00.0",f,lambda _:None)
            self.assertNotIn("scenes",events);self.assertIs(w._prewarm_cuda_for_curobo,orig)
    def test_native_exception_propagates(self):
        w,orig,f,_=self.harness(raises=True)
        with self.assertRaisesRegex(RuntimeError,"native failure"):delegate(w,ARGS,ENV,"0000:41:00.0",f,lambda _:None)
        self.assertIs(w._prewarm_cuda_for_curobo,orig)
    def test_pci_identity(self):
        self.assertEqual(pci("00000000:AB:00.0"),pci("0000:ab:00.0"))
        for value in ["ab:00",None,"0000:xx:00.0"]:
            with self.assertRaises(AssertionError):pci(value)
    def test_prewarm_failure_never_imports_renderer(self):
        w,orig,f,events=self.harness()
        def fail():raise RuntimeError("prewarm failure")
        w._prewarm_cuda_for_curobo=fail
        with self.assertRaisesRegex(RuntimeError,"prewarm failure"):delegate(w,ARGS,ENV,"0000:41:00.0",f,lambda _:None)
        self.assertNotIn("render_import_and_create",events);self.assertIs(w._prewarm_cuda_for_curobo,fail)
if __name__=="__main__":unittest.main()
