// CPU only: wrap all loader entry points; native GPU libraries are never linked.
#define NDEBUG
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#undef NDEBUG
#include <cassert>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
extern "C" int headless_init(),headless_support(),headless_loader(),headless_getproc(),headless_dispatcher(),headless_layer_symbol(),headless_layer_count(),headless_layer_data();
extern "C" unsigned headless_count();
static int scenario=0,open_calls=0,support_calls=0,enum_calls=0,proc_calls=0;
extern "C" int glfwInit(){return scenario==1||scenario==2?1:0;}
extern "C" int glfwVulkanSupported(){++support_calls;return scenario==2?1:0;}
static VkResult VKAPI_CALL fake_enum(uint32_t* count,VkLayerProperties* data){
 assert(count);++enum_calls;
 if(enum_calls==1){assert(!data);*count=scenario==8?0:scenario==9?4097:2;return scenario==10?VK_ERROR_INITIALIZATION_FAILED:VK_SUCCESS;}
 assert(enum_calls==2);assert(*count==(scenario==8?0u:2u));
 if(scenario!=8){assert(data);assert(data[0].layerName[0]==0);}
 if(scenario==11)return VK_INCOMPLETE;
 if(scenario==12)*count=3;
 return VK_SUCCESS;
}
static PFN_vkVoidFunction VKAPI_CALL fake_proc(VkInstance instance,const char*){assert(!instance);++proc_calls;return nullptr;}
extern "C" void* __wrap_dlopen(const char* name,int flags){
 assert(flags==(RTLD_NOW|RTLD_LOCAL));++open_calls;
 if(open_calls==1){assert(!std::strcmp(name,"libvulkan.so"));if(scenario==3||scenario==4||scenario==5)return nullptr;}
 else if(open_calls==2){assert(!std::strcmp(name,"libvulkan.so.1"));if(scenario==4||scenario==5)return nullptr;}
 else {assert(open_calls==3);assert(!std::strcmp(name,std::getenv("SAPIEN_VULKAN_LIBRARY_PATH")));}
 return reinterpret_cast<void*>(0x1234);
}
extern "C" int __wrap_dlclose(void* handle){assert(handle==reinterpret_cast<void*>(0x1234));return 0;}
extern "C" void* __wrap_dlsym(void* handle,const char* name){
 assert(handle==reinterpret_cast<void*>(0x1234));
 if(!std::strcmp(name,"vkGetInstanceProcAddr"))return scenario==6?nullptr:reinterpret_cast<void*>(fake_proc);
 assert(!std::strcmp(name,"vkEnumerateInstanceLayerProperties"));assert(proc_calls>0);
 return scenario==7?nullptr:reinterpret_cast<void*>(fake_enum);
}
int main(int argc,char** argv){
 assert(argc==2);scenario=std::atoi(argv[1]);
 int(*steps[])()={headless_init,headless_support,headless_loader,headless_getproc,headless_dispatcher,headless_layer_symbol,headless_layer_count,headless_layer_data};
 if(scenario>=20){int target=scenario-20;for(int i=0;i<target;++i)assert(steps[i]()>=0);assert(steps[target==7?0:target+1]()==-1);for(auto fn:steps)assert(fn()==-1);}
 else {
 bool failed=false;for(auto fn:steps){if(fn()==-1){failed=true;break;}}
 assert(failed==(scenario==2||scenario==4||scenario==6||scenario==7||scenario>=9));
 assert(support_calls==(scenario==1||scenario==2?1:0));
 if(!failed){assert(enum_calls==2);assert(headless_count()==(scenario==8?0u:2u));}
 int prior=open_calls;for(auto fn:steps)assert(fn()==-1);assert(open_calls==prior);
 }
 std::printf("scenario %d passed; native driver calls0\n",scenario);
}
