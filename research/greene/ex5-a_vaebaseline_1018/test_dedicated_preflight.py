"""Compare the new CPU route against frozen numerical gates, without GPU work."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import report_complete as report

SOURCE=Path(__file__).parent
ORIGINAL=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/code')

def function(source,name):
    return next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)

class DedicatedPreflightTests(unittest.TestCase):
    def test_entire_gpu_stage_body_and_caps_are_identical(self):
        old=(ORIGINAL/'run_preflight.py').read_text();new=(SOURCE/'dedicated_preflight.py').read_text()
        marker="        state['capacity_sample']=samples;started=time.monotonic();"
        old=old[old.index(marker):old.index("\n\nif __name__")]
        new=new[new.index(marker):new.index("\n\nif __name__")]
        self.assertEqual(old,new.replace("ROOT/'dedicated-preflight-logs'","ROOT/'preflight-logs'"))
        module=ast.parse((SOURCE/'dedicated_preflight.py').read_text())
        source_assignment=next(x for x in module.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE' for t in x.targets))
        self.assertEqual(ast.unparse(source_assignment.value),"ROOT / 'code'")

    def test_release_scientific_predicates_are_identical(self):
        old=function((ORIGINAL/'release_training.py').read_text(),'readiness')
        text=(SOURCE/'dedicated_release_training.py').read_text().replace("ROOT/'dedicated-preflight-run.json'","ROOT/'preflight-run.json'")
        new=function(text,'readiness')
        new.body=[x for x in new.body if not (isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['plan','assignment'] for t in x.targets))
            and not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and ast.unparse(x.value.func)=='j.legacy_failed')
            and not (isinstance(x,ast.Assert) and "preflight['resource_plan_sha256']" in ast.unparse(x))]
        new.body[-1].value.keywords=[x for x in new.body[-1].value.keywords if x.arg not in ['preflight_state_path','joint_plan_path','joint_plan_sha256','reserved_gpus']]
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_report_selects_hashed_new_state_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=root/'joint-allocation-plan.json';assignment=root/'joint-allocation.json'
            plan.write_text('{}');assignment.write_text('{}')
            state=root/'dedicated-preflight-run.json';state.write_text(json.dumps(dict(resource_plan_sha256=report.sha(plan),assignment_sha256=report.sha(assignment))))
            release=dict(preflight_state_path=str(state),joint_plan_sha256=report.sha(plan),evidence_sha256={str(state):report.sha(state)})
            self.assertEqual(report.selected_preflight(root,release),state)
            state.write_text('{}')
            with self.assertRaises(AssertionError):report.selected_preflight(root,release)
            release['preflight_state_path']=str(root/'unregistered.json')
            with self.assertRaises(AssertionError):report.selected_preflight(root,release)

    def test_report_requires_explicit_registered_v2_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=root/'joint-allocation-plan-v2.json';assignment=root/'joint-allocation.json'
            plan.write_text('{"version":2}');assignment.write_text('{}')
            state=root/'dedicated-preflight-run.json';state.write_text(json.dumps(dict(resource_plan_sha256=report.sha(plan),assignment_sha256=report.sha(assignment))))
            release=dict(preflight_state_path=str(state),joint_plan_path=str(plan),joint_plan_sha256=report.sha(plan),evidence_sha256={str(state):report.sha(state)})
            self.assertEqual(report.selected_preflight(root,release),state)
            plan.write_text('changed')
            with self.assertRaises(AssertionError):report.selected_preflight(root,release)
            release['joint_plan_path']=str(root/'unregistered-plan.json')
            with self.assertRaises(AssertionError):report.selected_preflight(root,release)

    def test_report_preserves_legacy_path_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'preflight-run.json';state.write_text('{}')
            self.assertEqual(report.selected_preflight(root,dict(evidence_sha256={str(state):report.sha(state)})),state)

if __name__=='__main__':unittest.main()
