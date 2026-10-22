// CPU-only executable: --wrap prevents every real dlopen/dlsym call.
#include <vulkan/vulkan_core.h>
#include <dlfcn.h>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
extern "C" int loader_split_open(),loader_split_symbol(),loader_split_proc(),loader_split_count_query(),loader_split_data_query();
extern "C" unsigned loader_split_count();
extern "C" const char* loader_split_error();
static int scenario=0,calls=0;
static VkResult VKAPI_CALL fake_enum(const char* layer,uint32_t* count,VkExtensionProperties* data){
 assert(layer==nullptr && count!=nullptr);
 if(!data){assert(calls++==3);if(scenario==4)return VK_ERROR_INITIALIZATION_FAILED;*count=scenario==6?0:scenario==7?4097:2;return VK_SUCCESS;}
 assert(calls++==4 && *count==2);
 assert(data[0].extensionName[0]==0 && data[1].specVersion==0);
 std::strcpy(data[0].extensionName,"VK_KHR_surface");
 if(scenario==5)return VK_INCOMPLETE;
 if(scenario==8)*count=3;
 return VK_SUCCESS;
}
static PFN_vkVoidFunction VKAPI_CALL fake_proc(VkInstance instance,const char* name){
 assert(calls++==2 && instance==VK_NULL_HANDLE);
 assert(std::strcmp(name,"vkEnumerateInstanceExtensionProperties")==0);
 return scenario==3?nullptr:reinterpret_cast<PFN_vkVoidFunction>(fake_enum);
}
extern "C" void* __wrap_dlopen(const char* name,int flags){
 assert(calls++==0 && std::strcmp(name,"libvulkan.so.1")==0 && flags==1);
 return scenario==1?nullptr:reinterpret_cast<void*>(0x1234);
}
extern "C" void* __wrap_dlsym(void* handle,const char* name){
 assert(calls++==1 && handle==reinterpret_cast<void*>(0x1234));
 assert(std::strcmp(name,"vkGetInstanceProcAddr")==0);
 return scenario==2?nullptr:reinterpret_cast<void*>(fake_proc);
}
int main(int argc,char** argv){
 assert(argc==2);scenario=std::atoi(argv[1]);
 int(*steps[])()={loader_split_open,loader_split_symbol,loader_split_proc,loader_split_count_query,loader_split_data_query};
 if(scenario>=10 && scenario<=14){
  int stop=scenario-10;
  for(int i=0;i<stop;++i)assert(steps[i]()==1);
  assert(steps[stop==4?0:stop+1]()==-1);
  int prior=calls;
  for(auto fn:steps)assert(fn()==-1);
  assert(calls==prior);
 } else {
  bool failed=false;
  for(auto fn:steps){int result=fn();if(result==-1){failed=true;break;}assert(result==1);}
  assert(failed==(scenario!=0));
  int prior=calls;
  for(auto fn:steps)assert(fn()==-1);
  assert(calls==prior);
  if(scenario==0)assert(loader_split_count()==2 && calls==5);
 }
 std::printf("mock scenario %d passed; real driver calls=0\n",scenario);
}
