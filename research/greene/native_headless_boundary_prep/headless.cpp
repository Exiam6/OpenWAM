// Diagnostic only. Production code must never execute outside a frozen dispatcher job.
#ifndef NDEBUG
#error Native dispatcher ABI requires release-mode headers; debug layout is incompatible
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <cstdlib>
#include <cstddef>
static_assert(sizeof(vk::DispatchLoaderDynamic)==0x13d8, "Installed ELF dispatcher size");
static_assert(offsetof(vk::DispatchLoaderDynamic,vkGetInstanceProcAddr)==0x48, "Native getproc offset");
static_assert(offsetof(vk::DispatchLoaderDynamic,vkCreateInstance)==0, "Native create offset");
static_assert(offsetof(vk::DispatchLoaderDynamic,vkEnumerateInstanceExtensionProperties)==0x68, "Native extensions offset");
static_assert(offsetof(vk::DispatchLoaderDynamic,vkEnumerateInstanceLayerProperties)==0x78, "Native layers offset");
static_assert(offsetof(vk::DispatchLoaderDynamic,vkEnumerateInstanceVersion)==0x448, "Native version offset");
#include <filesystem>
#include <memory>
#include <vector>
extern "C" int glfwInit();
extern "C" int glfwVulkanSupported();
namespace {
int phase=0, init_result=0, raw_result=0;
uint32_t count=0;
const char* error="";
std::unique_ptr<vk::DynamicLoader> loader;
PFN_vkGetInstanceProcAddr getproc=nullptr;
PFN_vkEnumerateInstanceLayerProperties enumerate=nullptr;
bool enter(int expected){int prior=phase;phase=-1;if(prior!=expected){error="Invalid order; never retry";return false;}return true;}
int fail(const char* msg){error=msg;return -1;}
}
extern "C" const char* headless_error(){return error;}
extern "C" int headless_raw_result(){return raw_result;}
extern "C" unsigned headless_count(){return count;}
extern "C" int headless_init(){
 if(!enter(0))return -1;
 init_result=glfwInit();phase=1;return init_result;
}
extern "C" int headless_support(){
 if(!enter(1))return -1;
 int supported=init_result?glfwVulkanSupported():0;
 if(supported)return fail("Display-capable branch outside headless diagnostic");
 phase=2;return 0;
}
extern "C" int headless_loader(){
 if(!enter(2))return -1;
 try{loader=std::make_unique<vk::DynamicLoader>();}catch(std::runtime_error&){loader.reset();}
 if(!loader){
  auto path=std::getenv("SAPIEN_VULKAN_LIBRARY_PATH");
  if(path && std::filesystem::is_regular_file(std::filesystem::path(path))){
   try{loader=std::make_unique<vk::DynamicLoader>(path);}catch(std::runtime_error&){loader.reset();}
  }
 }
 if(!loader)return fail("Native loader unavailable");
 phase=3;return 1;
}
extern "C" int headless_getproc(){
 if(!enter(3))return -1;
 getproc=loader->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
 if(!getproc)return fail("Missing vkGetInstanceProcAddr");
 phase=4;return 1;
}
extern "C" int headless_dispatcher(){
 if(!enter(4))return -1;
 VULKAN_HPP_DEFAULT_DISPATCHER.init(getproc);
 phase=5;return 1;
}
extern "C" int headless_layer_symbol(){
 if(!enter(5))return -1;
 enumerate=loader->getProcAddress<PFN_vkEnumerateInstanceLayerProperties>("vkEnumerateInstanceLayerProperties");
 if(!enumerate)return fail("Missing layer enumeration symbol");
 phase=6;return 1;
}
extern "C" int headless_layer_count(){
 if(!enter(6))return -1;
 raw_result=enumerate(&count,nullptr);
 if(raw_result!=VK_SUCCESS)return fail("Layer count VkResult diagnostic stop");
 if(count>4096)return fail("Diagnostic allocation bound exceeded");
 phase=7;return 1;
}
extern "C" int headless_layer_data(){
 if(!enter(7))return -1;
 uint32_t capacity=count;
 std::vector<VkLayerProperties> layers(count);
 raw_result=enumerate(&count,layers.data());
 if(raw_result!=VK_SUCCESS || count>capacity)return fail("Layer data VkResult/capacity diagnostic stop");
 phase=8;return 1;
}
