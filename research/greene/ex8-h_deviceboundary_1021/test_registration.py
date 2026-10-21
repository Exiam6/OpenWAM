import unittest
import ast
from pathlib import Path
from register import accounting
from run_probe import peers
class Admission(unittest.TestCase):
 def test_phase_boundaries_are_complete(self):
  source=(Path(__file__).parent/"probe_candidate.py").read_text()
  tree=ast.parse(source)
  names=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="checkpoint"]
  self.assertEqual(len(names),8)
  self.assertEqual(set(names),{"before_prewarm","after_prewarm","after_sapien_import","after_explicit_Device","after_explicit_RenderSystem","after_Engine","after_SapienRenderer","after_default_scene"})
 def test_native_build_values_match_frozen_full_runner(self):
  from prewarm import native_environment
  text=Path("/home/zifanz4/openwam-runtime/ex8-d_nativefull_1021/code/run_full.py").read_text()
  tree=ast.parse(text)
  actual={k.arg: k.value.value for n in ast.walk(tree) if isinstance(n,ast.Call) for k in n.keywords if isinstance(k.value,ast.Constant) and isinstance(k.value.value,str)}
  env=native_environment("/home/zifanz4/openwam-runtime","/output")
  for key in ["TORCH_CUDA_ARCH_LIST","MAX_JOBS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:self.assertEqual(env[key],actual[key])
  self.assertIn("PATH=f'{ROOT}/bin:{R}/benchmark-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin'",text)
 def test_uncertain_peer_blocks(self):
  self.assertEqual(peers([dict(id="ex8f-native-render-probe",status="submission_uncertain")]),["ex8f-native-render-probe"])
 def test_own_reserved_claim_allowed(self):
  self.assertEqual(peers([dict(id="ex8h-native-device-boundary",status="reserved")]),[])
 def test_terminal_peer_allowed(self):
  self.assertEqual(peers([dict(id="ex8d-full",status="failed")]),[])
 def test_budget_rounds_up(self):
  c=accounting([dict(status="failed",finished="now",occupied_parent_seconds=1.1)],[dict(max_seconds=100)]*97)
  self.assertEqual(c["terminal_rounded_seconds"],2)
 def test_conditional_validation_and_boundary_are_reserved(self):
  held=[dict(max_seconds=100)]*97
  result=accounting([dict(status="failed",finished="now",occupied_parent_seconds=2)],held)
  self.assertEqual(result["total_reserved_seconds"],2+9700+18749+300+9000)
  self.assertEqual(result["conditional_native_validation_seconds"],9000)
 def test_budget_excess_rejected(self):
  with self.assertRaises(AssertionError):accounting([dict(status="failed",finished="now",occupied_parent_seconds=1)],[dict(max_seconds=20000)]*97)
 def test_nonterminal_rejected(self):
  with self.assertRaises(AssertionError):accounting([dict(status="running",occupied_parent_seconds=1)],[dict(max_seconds=100)]*97)
if __name__=="__main__":unittest.main()
