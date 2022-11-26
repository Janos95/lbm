#pragma once

#include "WebGpuHelpers.h"
#include "callbacks.h"
#include "dump_utils.h"
#include "wgpu.h"

#include <Math/Vec.h>
#include <Math/Tensor.h>

#include <span>
#include <vector>

namespace oak {

class Application {
 public:
  Application();

  ~Application();

  void loop();

  void frame();

  void updateLbm();

  void populateBuffers();

 private:
  void initializeWindow();
  void initializeWebGpu();
  void initializeLbm();

  GLFWwindow* m_window = nullptr;

  wgpu::Buffer m_index_buffer;
  wgpu::Buffer m_vertex_buffer;
  wgpu::Texture m_colormap;
  wgpu::Sampler m_sampler;
  wgpu::BindGroup m_bind_group;

  wgpu::ShaderModule m_shader;
  wgpu::Instance m_instance;
  wgpu::Adapter m_adapter;
  wgpu::Device m_device;
  wgpu::Surface m_surface;
  wgpu::RenderPipeline m_pipeline;
  wgpu::SwapChain m_swapchain;

  std::vector<double> m_phasefield;
};

}  // namespace oak