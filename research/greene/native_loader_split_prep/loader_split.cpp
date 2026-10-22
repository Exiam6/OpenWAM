// Preparation only: no native execution before separate frozen registration.
#include <vulkan/vulkan_core.h>
#include <dlfcn.h>
#include <cstdlib>
static_assert(sizeof(VkExtensionProperties)==260,"Installed native allocation ABI");
namespace {
int phase=0;
const char* error="";
void* loader=nullptr;
PFN_vkGetInstanceProcAddr getproc=nullptr;
PFN_vkEnumerateInstanceExtensionProperties enumerate=nullptr;
uint32_t count=0;
bool enter(int expected) {
 int prior=phase; phase=-1;
 if(prior!=expected) { error="Invalid boundary order; never retry"; return false; }
 return true;
}
int fail(const char* message) {error=message;return -1;}
}
extern "C" const char* loader_split_error(){return error;}
extern "C" unsigned loader_split_count(){return count;}
extern "C" int loader_split_open(){
 if(!enter(0)) return -1;
 loader=dlopen("libvulkan.so.1",RTLD_LAZY|RTLD_LOCAL);
 if(!loader) return fail("dlopen failed");
 phase=1;return 1;
}
extern "C" int loader_split_symbol(){
 if(!enter(1)) return -1;
 getproc=reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(loader,"vkGetInstanceProcAddr"));
 if(!getproc) return fail("dlsym failed");
 phase=2;return 1;
}
extern "C" int loader_split_proc(){
 if(!enter(2)) return -1;
 enumerate=reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(getproc(VK_NULL_HANDLE,"vkEnumerateInstanceExtensionProperties"));
 if(!enumerate) return fail("getInstanceProcAddr failed");
 phase=3;return 1;
}
extern "C" int loader_split_count_query(){
 if(!enter(3)) return -1;
 VkResult result=enumerate(nullptr,&count,nullptr);
 if(result!=VK_SUCCESS) return fail("extension count query failed");
 if(count==0 || count>4096) return fail("diagnostic count guard");
 phase=4;return 1;
}
extern "C" int loader_split_data_query(){
 if(!enter(4)) return -1;
 auto* data=static_cast<VkExtensionProperties*>(std::calloc(count,sizeof(VkExtensionProperties)));
 if(!data) return fail("extension allocation failed");
 uint32_t capacity=count;
 VkResult result=enumerate(nullptr,&count,data);
 std::free(data);
 if(result!=VK_SUCCESS || count>capacity) return fail("extension data query failed");
 phase=5;return 1;
}
