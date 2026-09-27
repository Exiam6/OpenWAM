"""Cross-check diagnostic labels with the actual native dataloader, and known rotations."""
import sys
from pathlib import Path
import h5py,numpy as np
from omegaconf import OmegaConf
from openwam.dataloader.registry import build_dataset
from action_diagnostics import errors
E=Path('/data02/zifanz4/openwam-experiments/endtoend-20260923');sys.path.insert(0,str(E/'OpenWAM'))
from benchmarks.utils import action_conversion
ckpt=list((E/'profiles-v3/pca/output').glob('*/config.yaml'))[0];cfg=OmegaConf.load(ckpt);data=build_dataset(cfg.dataloader)
for ds in data._sub_datasets:
 with h5py.File(ds._episode_files[0],'r') as f:
  raw=ds._read_raw_actions(f,0,2)
  for i in [0,1]:
   x=action_conversion.robotwin_endpose_to_eef20d(f['endpose/left_endpose'][i],f['endpose/right_endpose'][i],f['endpose/left_gripper'][i],f['endpose/right_gripper'][i]);np.testing.assert_allclose(x,raw[i],rtol=0,atol=1e-6)
x=np.array([0,0,0,1,0,0,0,1,0,0]*2,dtype=float);assert all(abs(v)<1e-10 for v in errors(x,x).values())
y=x.copy();y[3:9]=[0,1,0,-1,0,0];y[13:19]=y[3:9];assert abs(errors(x,y)['rotation_mean_deg']-90)<1e-10
y=x.copy();y[[0,1,2,10,11,12]]=1;assert errors(x,y)['translation_mse_m2']==1
y=x.copy();y[[9,19]]=1;assert errors(x,y)['gripper_mse_native_units']==1
print('PASS: native dataset label conversion acrossall3tasks; exact0/90degree rotation, metric units andgripper dimensions')
