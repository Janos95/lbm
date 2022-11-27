// Copyright 2022 The Dusk Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "wgpu.h"

namespace oak {

wgpu::Buffer createBufferFromData(const wgpu::Device& device,
                                  const std::string& label,
                                  const void* data,
                                  uint64_t size,
                                  wgpu::BufferUsage usage);

wgpu::ShaderModule createShaderModule(const wgpu::Device& device,
                                      const std::string& label,
                                      const char* src);

wgpu::ShaderModule create_shader_from_file(const wgpu::Device& device,
                                           const std::string& path,
                                           const std::string& label);

wgpu::Texture createTexture(const wgpu::Device& device,
                            const std::string& label,
                            wgpu::Extent3D extent,
                            wgpu::TextureFormat format,
                            wgpu::TextureUsage usage);

wgpu::Texture createRgbaUnormTextureFromData(const wgpu::Device& device,
                                             size_t width,
                                             size_t height,
                                             const void* data);

// Helpers to make creating bind group layouts look nicer:
//
//   utils::MakeBindGroupLayout(device, {
//       {0, wgpu::ShaderStage::Vertex, wgpu::BufferBindingType::Uniform},
//       {1, wgpu::ShaderStage::Fragment, wgpu::SamplerBindingType::Filtering},
//       {3, wgpu::ShaderStage::Fragment, wgpu::TextureSampleType::Float}
//   });
struct BindingLayoutEntryInitializationHelper : wgpu::BindGroupLayoutEntry {
  BindingLayoutEntryInitializationHelper(uint32_t entryBinding,
                                         wgpu::ShaderStage entryVisibility,
                                         wgpu::BufferBindingType bufferType,
                                         bool bufferHasDynamicOffset = false,
                                         uint64_t bufferMinBindingSize = 0);

  BindingLayoutEntryInitializationHelper(uint32_t entryBinding,
                                         wgpu::ShaderStage entryVisibility,
                                         wgpu::SamplerBindingType samplerType);

  BindingLayoutEntryInitializationHelper(
      uint32_t entryBinding,
      wgpu::ShaderStage entryVisibility,
      wgpu::TextureSampleType textureSampleType,
      wgpu::TextureViewDimension viewDimension = wgpu::TextureViewDimension::e2D,
      bool textureMultisampled = false);

  BindingLayoutEntryInitializationHelper(
      uint32_t entryBinding,
      wgpu::ShaderStage entryVisibility,
      wgpu::StorageTextureAccess storageTextureAccess,
      wgpu::TextureFormat format,
      wgpu::TextureViewDimension viewDimension = wgpu::TextureViewDimension::e2D);

  BindingLayoutEntryInitializationHelper(uint32_t entryBinding,
                                         wgpu::ShaderStage entryVisibility,
                                         wgpu::ExternalTextureBindingLayout* bindingLayout);

  // NOLINTNEXTLINE(runtime/explicit)
  BindingLayoutEntryInitializationHelper(const wgpu::BindGroupLayoutEntry& entry);
};

wgpu::BindGroupLayout makeBindGroupLayout(
    const wgpu::Device& device,
    std::initializer_list<BindingLayoutEntryInitializationHelper> entriesInitializer);

// Helpers to make creating bind groups look nicer:
//
//   utils::MakeBindGroup(device, layout, {
//       {0, mySampler},
//       {1, myBuffer, offset, size},
//       {3, myTextureView}
//   });

// Structure with one constructor per-type of bindings, so that the initializer_list accepts
// bindings with the right type and no extra information.
struct BindingInitializationHelper {
  BindingInitializationHelper(uint32_t binding, const wgpu::Sampler& sampler);
  BindingInitializationHelper(uint32_t binding, const wgpu::TextureView& textureView);
  BindingInitializationHelper(uint32_t binding, const wgpu::ExternalTexture& externalTexture);
  BindingInitializationHelper(uint32_t binding,
                              const wgpu::Buffer& buffer,
                              uint64_t offset = 0,
                              uint64_t size = wgpu::kWholeSize);
  BindingInitializationHelper(const BindingInitializationHelper&) = default;
  ~BindingInitializationHelper() = default;

  wgpu::BindGroupEntry GetAsBinding() const;

  uint32_t binding;
  wgpu::Sampler sampler;
  wgpu::TextureView textureView;
  wgpu::Buffer buffer;
  wgpu::ExternalTextureBindingEntry externalTextureBindingEntry;
  uint64_t offset = 0;
  uint64_t size = 0;
};

wgpu::BindGroup makeBindGroup(
    const wgpu::Device& device,
    const wgpu::BindGroupLayout& layout,
    std::initializer_list<BindingInitializationHelper> entriesInitializer);

}  // namespace oak
