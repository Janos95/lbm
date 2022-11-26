//
// Created by Janos Meny on 04.11.22.
//

#include "Application.h"
#include "ColorMaps.h"

#include <fmt/core.h>

#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>

namespace oak {

// Simulation parameters
constexpr size_t Nx = 400;    // resolution x-dir
constexpr size_t Ny = 100;    // resolution y-dir
constexpr double rho0 = 100;  // average density
constexpr double tau = 0.6;   // collision timesclae
constexpr size_t Nt = 4000;   // number of timesteps

// Lattice speeds / weights
constexpr size_t NL = 9;

// idxs = np.arange(NL)
size_t idxs[NL] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

// cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
constexpr double cxs[] = {0, 0, 1, 1, 1, 0, -1, -1, -1};
// cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
constexpr double cys[] = {0, 1, 1, 0, -1, -1, -1, 0, 1};

// weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
constexpr double weights[] = {4. / 9, 1. / 9,  1. / 36, 1. / 9, 1. / 36,
                              1. / 9, 1. / 36, 1. / 9,  1. / 36};

// window size
constexpr size_t kWidth = Nx;
constexpr size_t kHeight = Ny;

constexpr double pi = std::numbers::pi;

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
  //return vec4<f32>(0.5*(u + 1.0), 0, 0, 1);
  let c = textureSample(colormap, colormap_sampler, vec2<f32>(0.48*(u + 1.0) + 0.02, 0));
  //let c = textureSample(colormap, colormap_sampler, vec2<f32>(0.5*(u + 1.0), 0));
  return c;
}
)";

void Application::initializeWebGpu() {
  m_instance = wgpu::CreateInstance();

  // Get Adapter
  m_instance.RequestAdapter(
      nullptr,
      [](WGPURequestAdapterStatus, WGPUAdapter adapterIn, const char*, void* userdata) {
        *static_cast<wgpu::Adapter*>(userdata) = wgpu::Adapter::Acquire(adapterIn);
      },
      &m_adapter);

  DumpAdapter(m_adapter);

  // Get device
  m_device = m_adapter.CreateDevice();
  m_device.SetLabel("Primary Device");

  m_device.SetUncapturedErrorCallback(cb::Error, nullptr);
  m_device.SetDeviceLostCallback(cb::DeviceLost, nullptr);
  // Logging is enabled as soon as the callback is setup.
  m_device.SetLoggingCallback(cb::Logging, nullptr);

  DumpDevice(m_device);

  // Get surface
  m_surface = wgpu::glfw::CreateSurfaceForWindow(m_instance, m_window);

  // Setup swapchain
  wgpu::SwapChainDescriptor swapchainDesc{
      .usage = wgpu::TextureUsage::RenderAttachment,
      .format = wgpu::TextureFormat::BGRA8Unorm,
      .width = kWidth,
      .height = kHeight,
      .presentMode = wgpu::PresentMode::Mailbox,
  };
  m_swapchain = m_device.CreateSwapChain(m_surface, &swapchainDesc);

  // Shaders
  m_shader = createShaderModule(m_device, "Main Shader Module", kShader);

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

  wgpu::RenderPipelineDescriptor pipelineDesc{
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
  m_pipeline = m_device.CreateRenderPipeline(&pipelineDesc);

  auto bgl = m_pipeline.GetBindGroupLayout(0);

  std::vector<unsigned char> map;
  for (const unsigned char* c : Turbo) {
    map.push_back(c[0]);
    map.push_back(c[1]);
    map.push_back(c[2]);
    map.push_back(255);
  }
  m_colormap =
      createRgbaUnormTextureFromData(m_device, 256, 1, reinterpret_cast<const void*>(map.data()));

  wgpu::SamplerDescriptor samplerDesc{.magFilter = wgpu::FilterMode::Linear,
                                      .minFilter = wgpu::FilterMode::Linear};

  m_sampler = m_device.CreateSampler(&samplerDesc);

  m_bind_group = makeBindGroup(m_device, bgl, {{0, m_sampler}, {1, m_colormap.CreateView()}});
}

void Application::initializeWindow() {
  glfwSetErrorCallback([](int code, const char* message) {
    throw std::runtime_error(fmt::format("GLFW error: {} - {}", code, message));
  });

  if (glfwInit() == 0) {
    throw std::runtime_error("Failed to initialize GLFW.");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);

  m_window = glfwCreateWindow(kWidth, kHeight, "Tiasa", nullptr, nullptr);
  if (m_window == nullptr) {
    throw std::runtime_error("Failed to create window");
  }
}

static void meshgrid(Tensor& xs, Tensor& ys) {
  xs = Tensor({Nx, Ny}, 0);
  ys = Tensor({Nx, Ny}, 0);
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      xs(x, y) = double(x);
      ys(x, y) = double(y);
    }
  }
}

double square(double x) {
  return x * x;
}

void Application::initializeLbm() {
  // X, Y = np.meshgrid(range(Nx), range(Ny))
  Tensor X, Y;
  meshgrid(X, Y);

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
        m_F(x, y, l) = 1. + 0.01 * dist(gen);
        if (l == 3) {
          m_F(x, y, l) += 2. * (1. + 0.2 * cos(2. * pi * X(x, y) / Nx * 4));
        }
      }
    }
  }

  // rho = np.sum(F,2)
  m_rho = Tensor({Nx, Ny}, 0.);
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      double& sum = m_rho(x, y);
      sum = 0.;
      for (size_t l = 0; l < NL; ++l) {
        sum += m_F(x, y, l);
      }
    }
  }

  // for i in idxs:
  //   F[:,:,i] *= rho0 / rho
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      for (auto l : idxs) {
        m_F(x, y, l) *= rho0 / m_rho(x, y);
      }
    }
  }

  // Cylinder boundary
  // cylinder = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**2
  m_cylinder = Tensor({Nx, Ny});
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      m_cylinder(x, y) = square(X(x, y) - Nx / 4) + square(Y(x, y) - Ny / 2) < square(Ny / 4);
    }
  }
}

Application::Application() {
  initializeWindow();
  initializeWebGpu();
  initializeLbm();
}

Application::~Application() {
  glfwDestroyWindow(m_window);
  glfwTerminate();
}

void Application::loop() {
  while (glfwWindowShouldClose(m_window) == 0) {
    updateLbm();
    populateBuffers();
    frame();
    glfwPollEvents();
  }
}

void Application::updateLbm() {
  // # Simulation Main Loop
  // for it in range(Nt):
  //
  // Drift
  //   for i, cx, cy in zip(idxs, cxs, cys):
  //     F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
  //     F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      for (size_t l = 0; l < NL; ++l) {
        NOT_IMPLEMENTED;
      }
    }
  }

  //
  // # Set reflective boundaries
  //   bndryF = F[cylinder,:]
  //   bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
  //
  //  Calculate fluid variables
  //   rho = np.sum(F,2)
  //   ux  = np.sum(F*cxs,2) / rho
  //   uy  = np.sum(F*cys,2) / rho
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      double& rho = m_rho(x, y);
      double& ux = m_ux(x, y);
      double& uy = m_uy(x, y);
      rho = 0.;
      for (size_t l = 0; l < NL; ++l) {
        rho += m_F(x, y, l);
        ux += m_F(x, y, l) * cxs[l];
        uy += m_F(x, y, l) * cys[l];
      }
      for (size_t l = 0; l < NL; ++l) {
      }
      ux /= rho;
      uy /= rho;
    }
  }

  // # Apply Collision
  //   Feq = np.zeros(F.shape)
  //   for i, cx, cy, w in zip(idxs, cxs, cys, weights):
  //     Feq[:,:,i] = rho*w* (1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      double ux = m_ux(x, y);
      double uy = m_uy(x, y);
      double rho = m_rho(x, y);
      for (size_t l = 0; l < NL; ++l) {
        m_Feq(x, y, l) =
            rho * weights[l] *
            (1. + 3. * (cxs[l] * ux + cys[l] * uy) + 9 * square(cxs[l] * ux + cys[l] * uy) / 2 -
             3 * (square(ux) + square(uy)) / 2);
      }
    }
  }

  // F += -(1.0/tau) * (F - Feq)
  for (size_t x = 0; x < Nx; ++x) {
    for (size_t y = 0; y < Ny; ++y) {
      for (size_t l = 0; l < NL; ++l) {
        m_F(x, y, l) += -(1. / tau) * (m_F(x, y, l) - m_Feq(x, y, l));
      }
    }
  }

  // # Apply boundary
  for (auto [x, y] : m_cylinder) {
    for (size_t l = 0; l < NL; ++l) {
      m_F(x, y, l) = boundaryF[i];
    }
  }
  //   F[cylinder,:] = bndryF
}

void Application::populateBuffers() {
  // Create buffers
  m_index_buffer =
      createBufferFromData(m_device, "Index Buffer", m_index_data.data(),
                           m_index_data.size() * sizeof(uint32_t), wgpu::BufferUsage::Index);
  m_vertex_buffer =
      createBufferFromData(m_device, "Vertex Buffer", m_vertex_data.data(),
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

    wgpu::RenderPassDescriptor renderPass{
        .label = "Main Render Pass",
        .colorAttachmentCount = 1,
        .colorAttachments = &attachment,
    };

    auto pass = encoder.BeginRenderPass(&renderPass);
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
