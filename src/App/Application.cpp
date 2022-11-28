//
// Created by Janos Meny on 04.11.22.
//

#include "Application.h"
#include "ColorMaps.h"

#include <fmt/chrono.h>
#include <fmt/core.h>

#include <chrono>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>

#include <tbb/parallel_for.h>

namespace oak {

// Simulation parameters
constexpr size_t Nx = 800;    // resolution x-dir
constexpr size_t Ny = 200;    // resolution y-dir
constexpr double rho0 = 100;  // average density
constexpr double tau = 0.6;   // collision timesclae

// Lattice speeds / weights
constexpr size_t NL = 9;

constexpr int cxs[] = {0, 0, 1, 1, 1, 0, -1, -1, -1};
constexpr int cys[] = {0, 1, 1, 0, -1, -1, -1, 0, 1};

constexpr double weights[] = {4. / 9, 1. / 9,  1. / 36, 1. / 9, 1. / 36,
                              1. / 9, 1. / 36, 1. / 9,  1. / 36};

// window size
constexpr size_t kWidth = 2 * Nx;
constexpr size_t kHeight = 2 * Ny;

// constexpr double pi = std::numbers::pi;

constexpr const char* kShader = R"(
 struct VertexInput {
   @location(0) pos: vec4<f32>,
   @location(1) uv: vec4<f32>,
 }

 struct VertexOutput {
   @builtin(position) pos: vec4<f32>,
   @location(0) uv: vec4<f32>,
 }

@vertex
 fn vs_main(in : VertexInput) -> VertexOutput {
   return VertexOutput(in.pos, in.uv);
 }

@group(0) @binding(0) var colormap_sampler: sampler;
@group(0) @binding(1) var colormap : texture_2d<f32>;

@fragment
 fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
  let u = in.uv.x;
  let c = textureSample(colormap, colormap_sampler, vec2<f32>(5*u, 0));
  return c;
}
)";

void Application::prepare_compute_pipeline() {
  // Compute shader
  m_lbm_shader = create_shader_from_file(m_device, "lbm.wgsl", "lbm shader");
  m_lbm_descriptor = wgpu::ProgrammableStageDescriptor{
      .module = m_lbm_shader,
      .entryPoint = "main",
  };

  wgpu::ComputePipelineDescriptor descriptor{
      .label = "lbm_pipeline",
      .compute = m_lbm_descriptor,
  };

  // Compute pipeline
  m_lbm_pipeline = m_device.CreateComputePipeline(&descriptor);

  // blur_pipeline = wgpuDeviceCreateComputePipeline(
  //     wgpu_context->device, &(WGPUComputePipelineDescriptor){
  //                               .label = "image_blur_render_pipeline",
  //                               .compute = blur_comp_shader.programmable_stage_descriptor,
  //                           });
  //  ASSERT(blur_pipeline != NULL);

  // Partial clean-up
  // wgpu_shader_release(&blur_comp_shader);
}

//  auto backbufferView = m_swapchain.GetCurrentTextureView();
//  backbufferView.SetLabel("Back Buffer Texture View");

//  wgpu::RenderPassColorAttachment attachment{
//      .view = backbufferView,
//      .loadOp = wgpu::LoadOp::Clear,
//      .storeOp = wgpu::StoreOp::Store,
//      .clearValue = {0., 0., 0., 1.},
//  };

//  wgpu::RenderPassDescriptor renderPass{
//      .label = "Main Render Pass",
//      .colorAttachmentCount = 1,
//      .colorAttachments = &attachment,
//  };

//  auto pass = encoder.BeginRenderPass(&renderPass);
//  pass.SetPipeline(m_pipeline);
//  pass.SetVertexBuffer(0, m_vertex_buffer);
//  pass.SetIndexBuffer(m_index_buffer, wgpu::IndexFormat::Uint32);
//  pass.SetBindGroup(0, m_bind_group);
//  pass.DrawIndexed(uint32_t(m_index_data.size()));
//  pass.End();
//}

void Application::encode_compute_pass(wgpu::CommandEncoder& encoder) {
  // wgpu_context->cpass_enc = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
  auto pass = encoder.BeginComputePass();

  // wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc, blur_pipeline);
  pass.SetPipeline(m_lbm_pipeline);

  // wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0, compute_constants_bind_group, 0,
  //                                    NULL);
  pass.SetBindGroup(0, m_compute_constants_bind_group);

  // wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1, compute_bind_groups[0], 0,
  // NULL);
  pass.SetBindGroup(1, m_compute_bind_groups[0]);

  const uint32_t tile_dim = 128;
  const uint32_t filter_size = 15;
  const uint32_t block_dim = tile_dim - filter_size - 1;
  const uint32_t batch[2] = {4, 4};
  const uint32_t iterations = 2;

  const auto wg_count_x = uint32_t(std::ceil(float(Nx) / float(block_dim)));
  const auto wg_count_y = uint32_t(std::ceil(float(Ny) / float(batch[1])));

  // wgpuComputePassEncoderDispatchWorkgroups(wgpu_context->cpass_enc,
  //                                          ceil((float)image_width / block_dim),
  //                                          ceil((float)image_height / batch[1]), 1);
  pass.DispatchWorkgroups(wg_count_x, wg_count_y, 1);

  // wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1, compute_bind_groups[1], 0,
  // NULL);
  pass.SetBindGroup(0, m_compute_bind_groups[1]);

  // wgpuComputePassEncoderDispatchWorkgroups(wgpu_context->cpass_enc,
  //                                          ceil((float)image_height / block_dim),
  //                                          ceil((float)image_width / batch[1]), 1);
  pass.DispatchWorkgroups(wg_count_x, wg_count_y, 1);

  for (uint32_t i = 0; i < iterations; ++i) {
    // wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1, compute_bind_groups[2], 0,
    // NULL);
    pass.SetBindGroup(1, m_compute_bind_groups[2]);
    // wgpuComputePassEncoderDispatchWorkgroups(wgpu_context->cpass_enc,
    //                                          ceil((float)image_width / block_dim),
    //                                          ceil((float)image_height / batch[1]), 1);
    pass.DispatchWorkgroups(wg_count_x, wg_count_y, 1);

    // wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1, compute_bind_groups[1], 0,
    // NULL);
    pass.SetBindGroup(1, m_compute_bind_groups[1]);
    // wgpuComputePassEncoderDispatchWorkgroups(wgpu_context->cpass_enc,
    //                                          ceil((float)image_height / block_dim),
    //                                          ceil((float)image_width / batch[1]), 1);
    pass.DispatchWorkgroups(wg_count_x, wg_count_y, 1);
  }

  // wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
  pass.End();
  // WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
}

void Application::initialize_webgpu() {
  m_instance = wgpu::CreateInstance();

  // Get Adapter
  m_instance.RequestAdapter(
      nullptr,
      [](WGPURequestAdapterStatus, WGPUAdapter adapterIn, const char*, void* userdata) {
        *static_cast<wgpu::Adapter*>(userdata) = wgpu::Adapter::Acquire(adapterIn);
      },
      &m_adapter);

  // DumpAdapter(m_adapter);

  // Get device
  m_device = m_adapter.CreateDevice();
  m_device.SetLabel("Primary Device");

  m_device.SetUncapturedErrorCallback(cb::Error, nullptr);
  m_device.SetDeviceLostCallback(cb::DeviceLost, nullptr);
  // Logging is enabled as soon as the callback is setup.
  m_device.SetLoggingCallback(cb::Logging, nullptr);

  // DumpDevice(m_device);

  // Get surface
  m_surface = wgpu::glfw::CreateSurfaceForWindow(m_instance, m_window);

  // Setup swapchain
  wgpu::SwapChainDescriptor swapchain_descriptor{
      .usage = wgpu::TextureUsage::RenderAttachment,
      .format = wgpu::TextureFormat::BGRA8Unorm,
      .width = kWidth,
      .height = kHeight,
      .presentMode = wgpu::PresentMode::Mailbox,
  };
  m_swapchain = m_device.CreateSwapChain(m_surface, &swapchain_descriptor);

  // Shaders
  m_shader = create_shader_from_source(m_device, "Main Shader Module", kShader);

  // Pipeline creation
  wgpu::VertexAttribute vertAttributes[2] = {{
                                                 .format = wgpu::VertexFormat::Float32x4,
                                                 .offset = 0,
                                                 .shaderLocation = 0,
                                             },
                                             {
                                                 .format = wgpu::VertexFormat::Float32x4,
                                                 .offset = 4 * sizeof(float),
                                                 .shaderLocation = 1,
                                             }};

  wgpu::VertexBufferLayout vertBufferLayout{
      .arrayStride = 8 * sizeof(float),
      .attributeCount = 2,
      .attributes = static_cast<wgpu::VertexAttribute*>(vertAttributes),
  };

  wgpu::ColorTargetState target{
      .format = wgpu::TextureFormat::BGRA8Unorm,
      //.format = wgpu::TextureFormat::RGBA8Unorm,
  };

  wgpu::FragmentState fragState{
      .module = m_shader,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &target,
  };

  wgpu::RenderPipelineDescriptor pipeline_descriptor{
      .label = "Main Render Pipeline",
      .layout = nullptr,  // Automatic layout
      .vertex =
          {
              .module = m_shader,
              .entryPoint = "vs_main",
              .bufferCount = 1,
              .buffers = &vertBufferLayout,
          },
      .fragment = &fragState,
  };
  m_pipeline = m_device.CreateRenderPipeline(&pipeline_descriptor);

  auto bgl = m_pipeline.GetBindGroupLayout(0);

  std::vector<unsigned char> map;
  for (const unsigned char* c : Turbo) {
    map.push_back(c[0]);
    map.push_back(c[1]);
    map.push_back(c[2]);
    map.push_back(255);
  }
  m_colormap = create_rgba_unorm_texture_from_data(m_device, 256, 1,
                                                   reinterpret_cast<const void*>(map.data()));

  wgpu::SamplerDescriptor samplerDesc{.magFilter = wgpu::FilterMode::Linear,
                                      .minFilter = wgpu::FilterMode::Linear};

  m_sampler = m_device.CreateSampler(&samplerDesc);

  m_bind_group = make_bind_group(m_device, bgl, {{0, m_sampler}, {1, m_colormap.CreateView()}});
}

void Application::initialize_window() {
  glfwSetErrorCallback([](int code, const char* message) {
    throw std::runtime_error(fmt::format("GLFW error: {} - {}", code, message));
  });

  if (glfwInit() == 0) {
    throw std::runtime_error("Failed to initialize GLFW.");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);

  m_window = glfwCreateWindow(kWidth, kHeight, "LBM", nullptr, nullptr);
  if (m_window == nullptr) {
    throw std::runtime_error("Failed to create window");
  }
}

inline double square(double x) {
  return x * x;
}

void Application::reset_lbm() {
  // Initial Conditions - flow to the right with some perturbations
  // F = np.ones((Ny,Nx,NL)) + 0.01*np.random.randn(Ny,Nx,NL)
  // F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))

  m_F = Tensor({Ny, Nx, NL}, 1.);
  m_Feq = Tensor({Ny, Nx, NL}, 1.);

  std::default_random_engine gen(0);
  std::normal_distribution<double> dist;

  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      for (size_t l = 0; l < NL; ++l) {
        m_F(y, x, l) = 1. + 0.01 * dist(gen);
        if (l == 3) {
          m_F(y, x, l) = 2.3;
        }
      }
    }
  }

  // rho = np.sum(F,2)
  m_rho = Tensor({Ny, Nx}, 0.);
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      double& sum = m_rho(y, x);
      sum = 0.;
      for (size_t l = 0; l < NL; ++l) {
        sum += m_F(y, x, l);
      }
    }
  }

  // for i in idxs:
  //   F[:,:,i] *= rho0 / rho
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      for (size_t l = 0; l < NL; ++l) {
        m_F(y, x, l) *= rho0 / m_rho(y, x);
      }
    }
  }

  // Cylinder boundary
  // cylinder = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**2
  m_cylinder = Tensor({Ny, Nx});
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      bool is_inside = square(double(x) - Nx / 4.) + square(double(y) - Ny / 2.) < square(Ny / 4.);
      m_cylinder(y, x) = double(is_inside);
    }
  }

  m_rho = Tensor({Ny, Nx});
  m_ux = Tensor({Ny, Nx});
  m_uy = Tensor({Ny, Nx});

  //m_F = Tensor::from_file("/Users/janos/lbm/src/App/initial.txt", {Ny, Nx, NL});
}

void Application::update_lbm() {
  // 50111536.24357496
  // auto compare_tensors = [](const auto& t1, const auto& t2){
  // auto s = t1.size();
  // ASSERT(s == t2.size());
  // double max_d = 0;
  // for(size_t i = 0; i < s; ++i){
  //   auto d = std::abs(t1[i] - t2[i]);
  //   bool is_ok =  d < 1e-8;
  //   ASSERT(is_ok);
  //   max_d = std::max(max_d, d);
  // }
  // return max_d;
  //};

  //{
  //  auto initial = Tensor::from_file("/Users/janos/lbm/src/App/initial.txt", {Ny, Nx, NL});
  //  auto d = compare_tensors(initial, m_F);
  //  fmt::print("initial max d {}\n", d);
  //}

  // # Simulation Main Loop
  // for it in range(Nt):

  // F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
  // F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

  for (size_t y = 0; y < Ny; ++y) {
    for (size_t l : {6, 7, 8}) {
      m_F(y, Nx - 1, l) = m_F(y, Nx - 2, l);
    }
    for (size_t l : {2, 3, 4}) {
      m_F(y, size_t(0), l) = m_F(y, size_t(1), l);
    }
  }

  //
  // Drift
  //   for i, cx, cy in zip(idxs, cxs, cys):
  //     F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
  //     F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

  auto role = [](int i, int N) {
    if (i >= N)
      return 0;
    if (i < 0)
      return N - 1;
    return i;
  };

  auto copy = m_F;
  tbb::parallel_for(size_t(0), Nx, [&](size_t x) {
    // for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      for (size_t l = 0; l < NL; ++l) {
        int x1 = role(int(x) - cxs[l], Nx);
        int y1 = role(int(y) - cys[l], Ny);
        copy(y, x, l) = m_F(size_t(y1), size_t(x1), l);
      }
    }
  });
  //}
  m_F = copy;

  //{
  //  auto drift = Tensor::from_file("/Users/janos/lbm/src/App/drift.txt", {Ny, Nx, NL});
  //  auto d = compare_tensors(drift, m_F);
  //  fmt::print("max d for drift {}\n", d);
  //}

  // # Set reflective boundaries
  //   bndryF = F[cylinder,:]
  //   bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]

  auto bndryF = m_F;
  tbb::parallel_for(size_t(0), Nx, [&](size_t x) {
    // for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      if (m_cylinder(y, x) != 0.) {
        constexpr size_t map[] = {0, 5, 6, 7, 8, 1, 2, 3, 4};
        for (size_t l = 0; l < NL; ++l) {
          bndryF(y, x, l) = m_F(y, x, map[l]);
        }
      }
    }
  });
  //}

  //  Calculate fluid variables
  //   rho = np.sum(F,2)
  //   ux  = np.sum(F*cxs,2) / rho
  //   uy  = np.sum(F*cys,2) / rho
  double max_velocity = 0;

  tbb::parallel_for(size_t(0), Nx, [&](size_t x) {
    // for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      double& rho = m_rho(y, x);
      double& ux = m_ux(y, x);
      double& uy = m_uy(y, x);
      rho = 0.;
      ux = 0.;
      uy = 0.;
      for (size_t l = 0; l < NL; ++l) {
        rho += m_F(y, x, l);
        ux += m_F(y, x, l) * double(cxs[l]);
        uy += m_F(y, x, l) * double(cys[l]);
      }
      ux /= rho;
      uy /= rho;
      max_velocity = std::max(max_velocity, Vec2(ux, uy).norm());
    }
  });
  //}
  // printf("max vel: %f\n", max_velocity);

  // # Apply Collision
  //   Feq = np.zeros(F.shape)
  //   for i, cx, cy, w in zip(idxs, cxs, cys, weights):
  //     Feq[:,:,i] = rho*w* (1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)

  tbb::parallel_for(size_t(0), Nx, [&](size_t x) {
    // for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      Vec2 u(m_ux(y, x), m_uy(y, x));
      double rho = m_rho(y, x);
      double uu = dot(u, u);
      for (size_t l = 0; l < NL; ++l) {
        Vec2 c(cxs[l], cys[l]);
        double uc = dot(u, c);
        double uc2 = uc * uc;
        m_Feq(y, x, l) = rho * weights[l] * (1. + 3. * uc + 9. * uc2 / 2. - 3. * uu / 2.);
      }
    }
  });
  //}

  // F += -(1.0/tau) * (F - Feq)
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      for (size_t l = 0; l < NL; ++l) {
        m_F(y, x, l) += -(1. / tau) * (m_F(y, x, l) - m_Feq(y, x, l));
      }
    }
  }

  //{
  //  auto collision = Tensor::from_file("/Users/janos/lbm/src/App/collision.txt", {Ny, Nx, NL});
  //  auto d = compare_tensors(collision, m_F);
  //  fmt::print("max d for collision {}\n", d);
  //}

  // # Apply boundary
  //   F[cylinder,:] = bndryF
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      if (m_cylinder(y, x) == 0.) {
        continue;
      }
      for (size_t l = 0; l < NL; ++l) {
        m_F(y, x, l) = bndryF(y, x, l);
      }
    }
  }

  //{
  //  auto boundary = Tensor::from_file("/Users/janos/lbm/src/App/boundary.txt", {Ny, Nx, NL});
  //  auto d = compare_tensors(boundary, m_F);
  //  fmt::print("max d for boundary {}\n", d);
  //}
}

Application::Application() {
  initialize_window();
  initialize_webgpu();
  reset_lbm();
}

Application::~Application() {
  glfwDestroyWindow(m_window);
  glfwTerminate();
}

inline void set_window_fps(GLFWwindow* win) {
  static int nb_frames = 0;
  static auto last_time = std::chrono::steady_clock::now();
  auto current_time = std::chrono::steady_clock::now();
  nb_frames++;

  using ms = std::chrono::milliseconds;
  auto dur = std::chrono::duration_cast<ms>(current_time - last_time);
  if (dur.count() >= 1000) {
    auto title = fmt::format("LBM - [Frame time: {}]", dur / nb_frames);
    glfwSetWindowTitle(win, title.c_str());
    nb_frames = 0;
    last_time = current_time;
  }
}

void Application::loop() {
  while (glfwWindowShouldClose(m_window) == 0) {
    set_window_fps(m_window);
    update_lbm();
    populate_buffers();
    frame();
    glfwPollEvents();
  }
}

inline void create_grid_mesh(Vec2 min,
                             Vec2 max,
                             Vec2u grid_size,
                             const Tensor& ux,
                             const Tensor& uy,
                             std::vector<float>& vertex_data,
                             std::vector<uint32_t>& index_data) {
  auto to1dFrom2d = [grid_size](size_t x, size_t y) { return uint32_t(x + y * grid_size.x); };

  auto rows = grid_size.y;
  auto columns = grid_size.x;
  vertex_data.reserve(rows * columns);
  for (size_t y = 0; y < rows; ++y) {
    for (size_t x = 0; x < columns; ++x) {
      auto px = (max.x - min.x) * double(x) / double(columns - 1) + min.x;
      auto py = (max.y - min.y) * double(y) / double(rows - 1) + min.y;
      auto u = Vec2(ux(y, x), uy(y, x)).norm();
      vertex_data.insert(vertex_data.end(), {float(px), float(py), 0.f, 1.f, float(u), 0, 0, 0});
    }
  }

  index_data.reserve((rows - 1) * (columns - 1) * 2);
  for (size_t y = 0; y < rows - 1; ++y) {
    for (size_t x = 0; x < columns - 1; ++x) {
      // (x,y+1) ---- (x+1,y+1)
      //   |          /   |
      //   |       /      |
      //   |    /         |
      //   | /            |
      // (x,y) ------- (x+1,y)
      auto v0 = to1dFrom2d(x, y), v1 = to1dFrom2d(x + 1, y), v2 = to1dFrom2d(x + 1, y + 1);
      auto w0 = to1dFrom2d(x, y), w1 = to1dFrom2d(x + 1, y + 1), w2 = to1dFrom2d(x, y + 1);
      index_data.insert(index_data.end(), {uint32_t(v0), uint32_t(v1), uint32_t(v2), uint32_t(w0),
                                           uint32_t(w1), uint32_t(w2)});
    }
  }
}

void Application::populate_buffers() {
  m_index_data.clear();
  m_vertex_data.clear();

  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      if (m_cylinder(y, x) != 0.) {
        m_ux(y, x) = 0.;
        m_uy(y, x) = 0.;
      }
    }
  }

  create_grid_mesh({-1, -1}, {1, 1}, {Nx, Ny}, m_ux, m_uy, m_vertex_data, m_index_data);

  // Create buffers
  m_index_buffer =
      create_buffer_from_data(m_device, "Index Buffer", m_index_data.data(),
                              m_index_data.size() * sizeof(uint32_t), wgpu::BufferUsage::Index);
  m_vertex_buffer =
      create_buffer_from_data(m_device, "Vertex Buffer", m_vertex_data.data(),
                              m_vertex_data.size() * sizeof(float), wgpu::BufferUsage::Vertex);
}

void Application::frame() {
  auto encoder = m_device.CreateCommandEncoder();
  encoder.SetLabel("Main Command Encoder");

  {
    auto backbufferView = m_swapchain.GetCurrentTextureView();
    backbufferView.SetLabel("Back Buffer Texture View");

    wgpu::RenderPassColorAttachment attachment{
        .view = backbufferView,
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0., 0., 0., 1.},
    };

    wgpu::RenderPassDescriptor descriptor{
        .label = "Main Render Pass",
        .colorAttachmentCount = 1,
        .colorAttachments = &attachment,
    };

    auto pass = encoder.BeginRenderPass(&descriptor);
    pass.SetPipeline(m_pipeline);
    pass.SetVertexBuffer(0, m_vertex_buffer);
    pass.SetIndexBuffer(m_index_buffer, wgpu::IndexFormat::Uint32);
    pass.SetBindGroup(0, m_bind_group);
    pass.DrawIndexed(uint32_t(m_index_data.size()));
    pass.End();
  }
  auto commands = encoder.Finish();

  m_device.GetQueue().Submit(1, &commands);
  m_swapchain.Present();
}

}  // namespace oak
