import numpy as np
from evaluate_policy import noise_observation
from analyze_closed_loop import paired_ci
raw=np.full((12,10,3),127,np.uint8);obs={'observation':{k:{'rgb':raw.copy()} for k in ['head_camera','left_camera','right_camera']},'state':np.arange(20)}
a=noise_observation(obs,.1,1100001,12);b=noise_observation(obs,.1,1100001,12);c=noise_observation(obs,.1,1100001,13)
assert a['state'] is obs['state']
for k in obs['observation']:
 assert np.array_equal(obs['observation'][k]['rgb'],raw)
 assert np.array_equal(a['observation'][k]['rgb'],b['observation'][k]['rgb'])
 assert not np.array_equal(a['observation'][k]['rgb'],c['observation'][k]['rgb'])
assert noise_observation(obs,0.,1100001,12) is obs
assert not np.array_equal(a['observation']['head_camera']['rgb'],a['observation']['left_camera']['rgb'])
z=paired_ci(np.zeros((3,3,4,50)),draws=1000);assert z['condition_95pp']==[[0.,0.]]*4 and z['perturbed_mean_95pp']==[0.,0.]
x=np.zeros((3,3,4,50));x[:,:,1:,:]=.2
z=paired_ci(x,draws=1000);np.testing.assert_allclose(z['point_pp'],[0,20,20,20],atol=1e-12);np.testing.assert_allclose(z['perturbed_mean_95pp'],[20,20],atol=1e-12)
# Training seed uncertainty must remain even with50 identical scenes per task.
x=np.broadcast_to(np.array([-.2,0,.2])[:,None,None,None],(3,3,4,50)).copy();z=paired_ci(x,draws=1000)
assert z['condition_95pp'][0][0]<0<z['condition_95pp'][0][1]
print('PASS: deterministic paired noise, original observations preserved, seed+scene bootstrap and constant/null contrasts')
