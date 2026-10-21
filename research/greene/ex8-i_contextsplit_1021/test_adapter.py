import copy
import types
import unittest
from adapter import delegate, guard, pci

UUID="GPU-89770bc8-d310-99d8-b3a4-d2b14c93d1a9"
ENV={"CUDA_VISIBLE_DEVICES":UUID,"OPENWAM_GPU_HOST":"gpu01","OPENWAM_ALLOCATION_END":"2026-10-21T23:00:00+00:00"}
ARGS=["--config","native.yml","--overrides","--task_name","adjust_bottle","--task_config","demo_clean","--ckpt_setting","ex8d-native-full","--seed","0","--policy_name","openwam2robotwin_interface"]
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
        for value in ["", "0", UUID+",GPU-other", "GPU-nope", "GPU-"+"-"*36]:
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
    def test_swallowed_prewarm_failure_cannot_import_sapien(self):
        import sys
        from unittest.mock import patch
        from adapter import native_render_factory
        bad_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
        with patch.dict(sys.modules,{"torch":bad_torch}):
            with self.assertRaises(AssertionError):native_render_factory(device="cuda:0")
            self.assertNotIn("sapien",sys.modules)
    def test_preserved_actual_native_argv(self):
        import json
        from pathlib import Path
        p=Path("/home/zifanz4/.local/state/openwam-selfcheck/evidence/20261021T2000Z-live-runtime-after-interruption.json")
        data=json.loads(p.read_text());seen=0
        for row in data["checks"]:
            for proc in row["processes"]:
                if proc["role"]=="benchmark":
                    guard(ENV,proc["argv"][2:]);seen+=1
        self.assertGreater(seen,0)
    def test_non_native_overrides_rejected(self):
        for flag,value in [("--seed","1"),("--policy_name","other"),("--task_config","custom")]:
            args=list(ARGS);args[args.index(flag)+1]=value
            with self.assertRaises(AssertionError):guard(ENV,args)
    def test_factory_cuda_before_sapien(self):
        import ast
        import inspect
        from adapter import native_render_factory
        tree=ast.parse(inspect.getsource(native_render_factory))
        torch_line=min(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Import) and any(a.name=="torch" for a in n.names))
        assert_line=min(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Assert))
        sapien_line=min(n.lineno for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module=="sapien.render")
        self.assertLess(torch_line,assert_line);self.assertLess(assert_line,sapien_line)
    def test_unregistered_probe_cannot_import_gpu(self):
        import json, tempfile, sys
        from pathlib import Path
        from probe_candidate import run
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"registration.json";out=Path(tmp)/"out.json"
            p.write_text(json.dumps({"kind":"FROZEN_NATIVE_PREWARM_AUDIT","registered_for_dispatch":False}))
            with self.assertRaises(AssertionError):run(p,out)
            self.assertFalse(out.exists())
            self.assertNotIn("sapien",sys.modules)
            self.assertNotIn("torch",sys.modules)
    def test_prewarm_failure_never_imports_renderer(self):
        w,orig,f,events=self.harness()
        def fail():raise RuntimeError("prewarm failure")
        w._prewarm_cuda_for_curobo=fail
        with self.assertRaisesRegex(RuntimeError,"prewarm failure"):delegate(w,ARGS,ENV,"0000:41:00.0",f,lambda _:None)
        self.assertNotIn("render_import_and_create",events);self.assertIs(w._prewarm_cuda_for_curobo,fail)
if __name__=="__main__":unittest.main()
