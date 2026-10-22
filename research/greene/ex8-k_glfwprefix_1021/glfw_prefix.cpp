// Diagnostic-only prefix; never load without frozen dispatcher authorization.
extern "C" int glfwInit();
extern "C" int glfwVulkanSupported();
namespace { int stage=0; const char *error=""; }
extern "C" const char *glfw_prefix_error() { return error; }
extern "C" int glfw_prefix_init() {
 int prior=stage; stage=-1;
 if(prior!=0) { error="Invalid init order; never retry"; return -1; }
 int result=glfwInit();
 if(result!=1) { error="Native glfwInit failed"; return result; }
 stage=1; return result;
}
extern "C" int glfw_prefix_supported() {
 int prior=stage; stage=-1;
 if(prior!=1) { error="Invalid support order; never retry"; return -1; }
 int result=glfwVulkanSupported();
 if(result!=1) { error="Native Vulkan support failed"; return result; }
 stage=2; return result;
}
