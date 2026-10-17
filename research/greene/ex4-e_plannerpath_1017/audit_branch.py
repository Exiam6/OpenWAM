"""Execute the real planner target transform without importing CUDA or simulator."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import transforms3d as t3d
import yaml

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-e_plannerpath_1017')
ROBOTWIN=ROOT.parent/'assets-source/benchmarks/RoboTwin'
OLD=ROOT.parent/'migration-20260927/robot-configs'
NEW=ROOT/'robot-configs/aloha-agilex'

def main():
    source=ROBOTWIN/'envs/robot/planner.py';tree=ast.parse(source.read_text())
    cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='CuroboPlanner')
    method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='plan_path'))
    end=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='goal_pose_of_ee' for t in n.targets))
    method.body=method.body[:end+1]+[ast.Return(value=ast.Name(id='goal_pose_of_ee',ctx=ast.Load()))]
    transform=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_trans_from_world_to_base'))
    shell=ast.ClassDef(name='AuditPlanner',bases=[],keywords=[],body=[method,transform],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[shell],type_ignores=[]))
    namespace=dict(np=np,t3d=t3d,CuroboPose=SimpleNamespace(from_list=lambda x:np.asarray(x)))
    exec(compile(module,str(source),'exec'),namespace)
    Planner=namespace['AuditPlanner'];rows=[];hashes={}
    for side,angle in [('left',-.02),('right',-.01)]:
        path=OLD/f'curobo_{side}.yml';alias=NEW/path.name
        assert 'aloha-agilex' not in str(path) and 'aloha-agilex' in str(alias)
        assert alias.read_bytes()==path.read_bytes()
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(alias)]=hashlib.sha256(alias.read_bytes()).hexdigest()
        cfg=yaml.safe_load(path.read_text());planner=Planner()
        planner.robot_origion_pose=SimpleNamespace(p=np.array([0.,-.65,.74]),q=np.array([1.,0.,0.,0.]))
        planner.frame_bias=cfg['planner']['frame_bias']
        target=SimpleNamespace(p=np.array([.12,-.10,.90]),q=np.array([1.,0.,0.,0.]))
        outputs={}
        for label,p in [('old',path),('alias',alias),('native',ROBOTWIN/'assets/embodiments/aloha-agilex'/path.name)]:
            planner.yml_path=str(p);outputs[label]=planner.plan_path([],target,arms_tag=side)
        np.testing.assert_array_equal(outputs['alias'],outputs['native'])
        R=t3d.axangles.axangle2mat([0,0,1],angle)
        np.testing.assert_allclose(outputs['alias'][:3],R@outputs['old'][:3],atol=1e-12,rtol=0)
        q=t3d.quaternions.quat2mat(outputs['alias'][3:])@t3d.quaternions.quat2mat(outputs['old'][3:]).T
        np.testing.assert_allclose(q,R,atol=1e-12,rtol=0)
        rows.append(dict(arm=side,calibration_rad=angle,calibration_degrees=float(np.degrees(angle)),
            same_yaml=True,alias_equals_native=True,example_position_difference_mm=float(1000*np.linalg.norm(outputs['alias'][:3]-outputs['old'][:3])),
            goals={k:v.tolist() for k,v in outputs.items()}))
    conditions=[ast.unparse(n.test) for n in ast.walk(cls) if isinstance(n,ast.If) and 'aloha-agilex' in ast.unparse(n.test)]
    assert len(conditions)==2
    result=dict(passed=True,planner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),config_sha256=hashes,
        conditional_sites=conditions,rows=rows,note='Exact goal-transform branch audit; no simulator-success conclusion.')
    (ROOT/'branch-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
