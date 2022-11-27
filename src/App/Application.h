#pragma once

#include "WebGpuHelpers.h"
#include "callbacks.h"
#include "dump_utils.h"
#include "wgpu.h"

#include <Math/Tensor.h>
#include <Math/Vec.h>

#include <span>
#include <vector>

namespace oak {

class Application {
 public:
  Application();

  ~Application();

  void loop();

  void frame();

  void update_lbm();

  void populate_buffers();

 private:
  void initialize_window();
  void initialize_webgpu();

  void prepare_compute_pipeline();
  void encode_compute_pass(wgpu::CommandEncoder&);

  void reset_lbm();

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

  // lbm compute pipeline
  wgpu::ShaderModule m_lbm_shader;
  wgpu::ProgrammableStageDescriptor m_lbm_descriptor;
  wgpu::ComputePipeline m_lbm_pipeline;
  wgpu::BindGroup m_compute_constants_bind_group;
  wgpu::BindGroup m_compute_bind_groups[3];

  std::vector<float> m_vertex_data;
  std::vector<uint32_t> m_index_data;

  Tensor m_F, m_Feq;
  Tensor m_rho;
  Tensor m_ux, m_uy;

  Tensor m_cylinder;
};

}  // namespace oak
