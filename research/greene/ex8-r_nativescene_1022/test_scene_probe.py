import math,types,unittest
from scene_probe import capture,drive
class Array(list):
 def __init__(self,values,shape,dtype='uint8',std=1):super().__init__(values);self.shape=shape;self.dtype=dtype;self._std=std
 def std(self):return self._std
 def tolist(self):return list(self)
def images():return {'observation':{n:{'rgb':Array([], (16,16,3))} for n in ['head_camera','left_camera','right_camera']}}
class SceneChecks(unittest.TestCase):
 def test_capture_native_shapes_and_finite_state(self):
  v=capture(images(),lambda _:Array([0]*20,(20,),'float32'),lambda n,r:n+'.png')
  self.assertEqual(len(v['state']),20);self.assertEqual(len(v['images']),3)
 def test_malformed_state_camera_and_blank_rejected(self):
  for state in [Array([0]*19,(19,)),Array([float('nan')]*20,(20,))]:
   with self.assertRaises(AssertionError):capture(images(),lambda _:state,lambda n,r:n)
  for shape,dtype,std in [((16,16,4),'uint8',1),((0,16,3),'uint8',1),((16,16,3),'float32',1),((16,16,3),'uint8',0)]:
   obs=images();obs['observation']['head_camera']['rgb']=Array([],shape,dtype,std)
   with self.subTest(shape=shape,dtype=dtype,std=std),self.assertRaises(AssertionError):capture(obs,lambda _:Array([0]*20,(20,)),lambda n,r:n)
 def fixture(self,fault=None):
  calls=[];device=types.SimpleNamespace(is_cuda=lambda:True,can_render=lambda:True,is_cpu=lambda:False,cuda_id=0,pci_string='0000:07:00.0',name='fake')
  env=types.SimpleNamespace(scene=types.SimpleNamespace(render_system=types.SimpleNamespace(device=device)))
  def setup(**kwargs):
   calls.append(('setup',kwargs))
   if fault=='setup':raise RuntimeError('native setup failure')
  env.setup_demo=setup;env.close_env=lambda:calls.append(('close',));env.get_obs=lambda:images()
  def forbid(*a):raise AssertionError('real policy function called')
  module=types.SimpleNamespace(eval_function_decorator=forbid,eval_policy=forbid)
  wrapper=types.SimpleNamespace(_prewarm_cuda_for_curobo=lambda:print('[eval_policy_wrapper] prewarmed CUDA/Curobo before SAPIEN import'))
  def bootstrap():wrapper._prewarm_cuda_for_curobo();return module
  wrapper.bootstrap_robotwin_module=bootstrap
  def main(argv):
   m=wrapper.bootstrap_robotwin_module();model=m.eval_function_decorator('policy','get_model')({})
   return m.eval_policy('adjust_bottle',env,{'task_config':'demo_clean'},model,100001 if fault=='seed' else 100000)
  wrapper.main=main
  return wrapper,module,env,calls,forbid,bootstrap
 def test_actual_callback_no_policy_cleanup_and_restore(self):
  w,m,e,c,original,b=self.fixture();proof=drive(w,[],'00000000:07:00.0','unused',lambda obs:{'valid':True})
  self.assertFalse(proof['policy_loaded']);self.assertFalse(proof['expert_rollout']);self.assertEqual(proof['scored_episodes'],0)
  self.assertEqual(c[0][1]['seed'],100000);self.assertTrue(c[0][1]['eval_mode']);self.assertEqual(c[-1],('close',))
  self.assertIs(m.eval_policy,original);self.assertIs(m.eval_function_decorator,original);self.assertIs(w.bootstrap_robotwin_module,b)
 def test_failure_and_wrong_seed_stop_without_hiding_error(self):
  for fault in ['setup','seed']:
   w,m,e,c,original,b=self.fixture(fault)
   with self.subTest(fault=fault),self.assertRaises((RuntimeError,AssertionError)):drive(w,[],'0000:07:00.0','unused',lambda _:None)
   self.assertIs(m.eval_policy,original);self.assertIs(w.bootstrap_robotwin_module,b)
   if fault=='setup':self.assertEqual(c[-1],('close',))
if __name__=='__main__':unittest.main()
