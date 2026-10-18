"""Predict replay position error independently from the URDF fixed-base yaw."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import transforms3d as t3d
import yaml

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-e_plannerpath_1017')
ASSET=ROOT.parent/'assets-source/benchmarks/RoboTwin/assets/embodiments/aloha-agilex'

def main():
    name='handover_block-seed4101100'
    target=np.load(ROOT/'code/demos'/name/'full-trajectory.npz')['proprio']
    config=yaml.safe_load((ASSET/'config.yml').read_text())
    origin=np.asarray(config['robot_pose'][0][:3]);R=t3d.quaternions.quat2mat(config['robot_pose'][0][3:])
    urdf=ET.parse(ASSET/'urdf/arx5_description_isaac.urdf').getroot()
    records=[]
    for condition in ['old','fixed']:
        path=ROOT/'replay'/condition/name/f'{name}-proprio.npy'
        if not path.exists():continue
        actual=np.load(path);desired=target[1:len(actual)+1]
        for side,link,xyz in [('left','fl_base_link',slice(0,3)),('right','fr_base_link',slice(10,13))]:
            joint=next(j for j in urdf.findall('joint') if j.find('child').get('link')==link)
            fixed=joint.find('origin');t=np.fromstring(fixed.get('xyz'),sep=' ');rpy=np.fromstring(fixed.get('rpy'),sep=' ')
            assert np.all(rpy[:2]==0)
            A=t3d.axangles.axangle2mat([0,0,1],rpy[2])
            want=desired[:,xyz]
            arm=(want-origin)@R-t
            expected=(arm@(A-np.eye(3)).T)@R.T
            measured=actual[:,xyz]-want
            quiet=np.zeros(len(want),dtype=bool)
            quiet[1:-1]=(np.linalg.norm(np.diff(want,axis=0)[:-1],axis=1)<1e-6)&(np.linalg.norm(np.diff(want,axis=0)[1:],axis=1)<1e-6)
            assert quiet.sum()>0
            error=np.linalg.norm(measured[quiet],axis=1)*1000
            residual=np.linalg.norm((measured-expected)[quiet],axis=1)*1000
            records.append(dict(condition=condition,arm=side,stationary_samples=int(quiet.sum()),
                all_samples=len(want),
                all_position_error_median_mm=float(np.median(np.linalg.norm(measured,axis=1)*1000)),
                all_missing_rotation_residual_median_mm=float(np.median(np.linalg.norm(measured-expected,axis=1)*1000)),
                all_missing_rotation_residual_p95_mm=float(np.quantile(np.linalg.norm(measured-expected,axis=1)*1000,.95)),
                urdf_base_yaw_rad=float(rpy[2]),position_error_median_mm=float(np.median(error)),
                position_error_p95_mm=float(np.quantile(error,.95)),
                missing_rotation_prediction_residual_median_mm=float(np.median(residual)),
                measured_mean_mm=(measured[quiet].mean(0)*1000).tolist(),
                predicted_missing_rotation_mean_mm=(expected[quiet].mean(0)*1000).tolist()))
    result=dict(records=records,note='Stationary target spans from the same replay; missing-rotation prediction comes from URDF and robot base pose, not fitted to measurements.')
    (ROOT/'geometry-crosscheck.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
