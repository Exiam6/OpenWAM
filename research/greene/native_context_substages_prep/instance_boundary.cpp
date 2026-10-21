// CPU-only preparation: compile/link only. Never load or invoke without a new registration.
#include <svulkan2/core/instance.h>
#include <svulkan2/core/physical_device.h>
#include <svulkan2/core/device.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdio>
namespace {
std::shared_ptr<svulkan2::core::Instance> instance;
std::shared_ptr<svulkan2::core::PhysicalDevice> physical;
std::shared_ptr<svulkan2::core::Device> logical;
std::string error;
int stage=0;
}
extern "C" const char *instance_boundary_error() { return error.c_str(); }
extern "C" int instance_boundary_create() {
 try {
  int prior=stage; stage=-1;
  if(prior!=0) throw std::runtime_error("Invalid Instance order; never retry");
  stage=-1;
  instance=svulkan2::core::Instance::Get(VK_MAKE_VERSION(0,0,1),VK_MAKE_VERSION(0,0,1),VK_API_VERSION_1_2);
  if(!instance) throw std::runtime_error("Native Instance failed");
  stage=1; return 0;
 } catch(std::exception const &e) { error=e.what(); return -1; }
}
extern "C" int instance_boundary_select() {
 try {
  int prior=stage; stage=-1;
  if(prior!=1) throw std::runtime_error("Invalid PhysicalDevice order; never retry");
  stage=-1;
  physical=instance->createPhysicalDevice("cuda:0");
  if(!physical) throw std::runtime_error("Native PhysicalDevice failed");
  stage=2; return 0;
 } catch(std::exception const &e) { error=e.what(); return -1; }
}
extern "C" int instance_boundary_logical() {
 try {
  int prior=stage; stage=-1;
  if(prior!=2) throw std::runtime_error("Invalid logical Device order; never retry");
  stage=-1;
  logical=physical->createDevice();
  if(!logical) throw std::runtime_error("Native logical Device failed");
  stage=3; return 0;
 } catch(std::exception const &e) { error=e.what(); return -1; }
}
// Deliberately stops before Context::Create, avoiding duplicate logical devices.
// Holding native shared_ptrs keeps sampled objects alive until process exit.

extern "C" const char *instance_boundary_selected() {
 static std::string selected;
 if(!physical || stage<2) { error="PhysicalDevice unavailable"; return nullptr; }
 auto const &info=physical->getPickedDeviceInfo();
 char pci[32]; std::snprintf(pci,sizeof(pci),"%04x:%02x:%02x.%x",info.pci[0],info.pci[1],info.pci[2],info.pci[3]);
 selected=std::string("{\"pci\":\"")+pci+"\",\"cuda_id\":"+std::to_string(info.cudaId)+",\"ray_tracing\":"+(info.rayTracing?"true":"false")+"}";
 return selected.c_str();
}
