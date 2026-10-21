// CPU-buildable diagnostic helper. Calling exported functions requires a separately
// registered dispatcher job; building this file does not create a GPU context.
#include <sapien/sapien_renderer/sapien_renderer_default.h>
#include <svulkan2/core/context.h>
#include <svulkan2/core/physical_device.h>
#include <memory>
#include <string>
#include <exception>
namespace {
std::shared_ptr<svulkan2::core::Context> context;
std::shared_ptr<svulkan2::resource::SVResourceManager> resources;
std::string error;
}
extern "C" const char *context_boundary_error() { return error.c_str(); }
extern "C" int context_boundary_create() {
  try {
    if (context) throw std::runtime_error("Context already created; never retry");
    using D = sapien::sapien_renderer::SapienRendererDefault;
    context = svulkan2::core::Context::Create(D::getMaxNumMaterials(),
       D::getMaxNumTextures(), D::getDefaultMipMaps(), D::getDoNotLoadTexture(), "cuda:0");
    if (!context || !context->isVulkanAvailable()) throw std::runtime_error("No native Vulkan context");
    return 0;
  } catch (std::exception const &e) { error=e.what(); return -1; }
}
extern "C" int context_boundary_resources() {
  try {
    if (!context || resources) throw std::runtime_error("Invalid resource boundary order");
    resources = context->createResourceManager();
    return resources ? 0 : -1;
  } catch (std::exception const &e) { error=e.what(); return -1; }
}
// Keep native objects alive through the monitored Python checkpoints. Process exit
// releases them; no independent context or alternate renderer is substituted.
