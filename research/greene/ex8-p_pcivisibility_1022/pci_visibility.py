"""Pure assignment-to-PCI selector. No native imports, no global changes."""
from native_adapter import pci
BLOCKED={'DRI_PRIME','NODEVICE_SELECT','MESA_VK_DEVICE_SELECT','MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE','VK_INSTANCE_LAYERS','VK_LAYER_PATH','VK_ADD_LAYER_PATH','VK_LOADER_LAYERS_DISABLE','VK_LOADER_LAYERS_ENABLE','VK_ICD_FILENAMES','VK_DRIVER_FILES','VK_ADD_DRIVER_FILES'}
def selector(expected):
 domain,bus,device,function=pci(expected)
 assert domain<=0xffff and bus<=0xff and device<=0x1f
 return f'pci-{domain:04x}_{bus:02x}_{device:02x}_{function}!'
def apply(env,expected):
 assert not BLOCKED.intersection(env),'Ambient Vulkan selection override'
 value=selector(expected);env['DRI_PRIME']=value
 return value
