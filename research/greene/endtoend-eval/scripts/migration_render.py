"""GPU placement only: preserve CPU physics and explicitly select the allocated renderer.

Copied from the GPU02 runtime-source; configure_robot_paths() is dropped because
Greene's RoboTwin embodiment configs were rewritten in place by
script/update_embodiment_config_path.py.
"""
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
