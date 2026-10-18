"""GPU placement only: preserve CPU physics and explicitly select the reserved renderer."""
import os

def pin_renderer():
    import torch
    assert torch.cuda.device_count()==1
    torch.zeros(1,device='cuda')
    from curobo.wrap.reacher.motion_gen import MotionGen
    import sapien
    from sapien.wrapper.scene import Scene
    from sapien.wrapper.renderer import SapienRenderer
    device=sapien.Device('cuda:0')
    from collect_scene import pci
    assert pci(device.pci_string)==pci(os.environ['EXPECTED_RENDER_PCI']),device.pci_string
    native_renderer=sapien.pysapien.render.SapienRenderer
    original_scene_init=Scene.__init__
    def renderer_init(self,**kwargs):
        native_renderer.__init__(self,device)
    def scene_init(self,systems=None):
        if systems is None:
            systems=[sapien.physx.PhysxCpuSystem(),sapien.render.RenderSystem(device)]
        original_scene_init(self,systems=systems)
    SapienRenderer.__init__=renderer_init
    Scene.__init__=scene_init
    print('RENDER_DEVICE_PINNED',device.pci_string,flush=True)


def remap_robot_config(path, config_root):
    """Relocate YAML without changing RoboTwin's path-selected calibration."""
    from pathlib import Path
    source = Path(path)
    root = Path(config_root)
    if source.name not in ('curobo_left.yml', 'curobo_right.yml'):
        return path
    target = root / source.name
    native_aloha = 'aloha-agilex' in str(source)
    if native_aloha and 'aloha-agilex' not in str(target):
        alias = root / 'aloha-agilex' / source.name
        alias.parent.mkdir(parents=True, exist_ok=True)
        try:
            alias.symlink_to(target.absolute())
        except FileExistsError:
            pass
        if alias.read_bytes() != target.read_bytes():
            raise ValueError(f'Existing robot-config alias differs: {alias}')
        target = alias
    if ('aloha-agilex' in str(target)) != native_aloha:
        raise ValueError(f'Relocation would change robot calibration: {source} -> {target}')
    # Do not resolve symlinks: planner.py uses the literal pathname in both
    # plan_path and plan_batch to select the ALOHA base-frame rotation.
    return str(target)


def configure_robot_paths():
    from pathlib import Path
    from envs.robot.planner import CuroboPlanner
    original = CuroboPlanner.__init__
    import os
    config_root=Path(os.environ.get('ROBOT_CONFIG_ROOT','/home/zifanz4/openwam-runtime/migration-20260927/robot-configs'))
    if not config_root.exists():
        print('configure_robot_paths: no remap (RoboTwin native curobo configs)',flush=True);return
    import inspect
    signature = inspect.signature(original)
    def initialize(self,*args,**kwargs):
        bound = signature.bind(self, *args, **kwargs)
        path = bound.arguments.get('yml_path')
        if path is not None:
            bound.arguments['yml_path'] = remap_robot_config(path, config_root)
        return original(*bound.args, **bound.kwargs)
    CuroboPlanner.__init__=initialize
