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

#include "WebGpuHelpers.h"

#include <vector>

namespace oak {

wgpu::Buffer createBufferFromData(const wgpu::Device& device,
                                  const std::string& label,
                                  const void* data,
                                  uint64_t size,
                                  wgpu::BufferUsage usage) {
  wgpu::BufferDescriptor desc{
      .label = label.c_str(),
      .usage = usage | wgpu::BufferUsage::CopyDst,
      .size = size,
  };
  auto buffer = device.CreateBuffer(&desc);
  device.GetQueue().WriteBuffer(buffer, 0, data, size);
  return buffer;
}

wgpu::ShaderModule createShaderModule(const wgpu::Device& device,
                                      const std::string& label,
                                      const char* src) {
  wgpu::ShaderModuleWGSLDescriptor wgslDesc;
  wgslDesc.source = src;
  wgpu::ShaderModuleDescriptor desc{
      .nextInChain = &wgslDesc,
      .label = label.c_str(),
  };
  auto shader = device.CreateShaderModule(&desc);
  return shader;
}

wgpu::Texture createTexture(const wgpu::Device& device,
                            const std::string& label,
                            wgpu::Extent3D extent,
                            wgpu::TextureFormat format,
                            wgpu::TextureUsage usage) {
  wgpu::TextureDescriptor desc{
      .label = label.c_str(),
      .usage = usage,
      .size = extent,
      .format = format,
  };

  return device.CreateTexture(&desc);
}

namespace {

wgpu::ImageCopyTexture createImageCopyTexture(
    wgpu::Texture texture,
    uint32_t level = 0,
    wgpu::Origin3D origin = {0, 0, 0},
    wgpu::TextureAspect aspect = wgpu::TextureAspect::All) {
  wgpu::ImageCopyTexture imageCopyTexture;
  imageCopyTexture.texture = texture;
  imageCopyTexture.mipLevel = level;
  imageCopyTexture.origin = origin;
  imageCopyTexture.aspect = aspect;

  return imageCopyTexture;
}

wgpu::TextureDataLayout createTextureDataLayout(uint64_t offset,
                                                uint32_t bytesPerRow,
                                                uint32_t rowsPerImage) {
  wgpu::TextureDataLayout textureDataLayout;
  textureDataLayout.offset = offset;
  textureDataLayout.bytesPerRow = bytesPerRow;
  textureDataLayout.rowsPerImage = rowsPerImage;

  return textureDataLayout;
}
wgpu::ImageCopyBuffer createImageCopyBuffer(wgpu::Buffer buffer,
                                            uint64_t offset,
                                            uint32_t bytesPerRow,
                                            uint32_t rowsPerImage) {
  wgpu::ImageCopyBuffer imageCopyBuffer = {};
  imageCopyBuffer.buffer = buffer;
  imageCopyBuffer.layout = createTextureDataLayout(offset, bytesPerRow, rowsPerImage);

  return imageCopyBuffer;
}

}  // namespace

wgpu::Texture createRgbaUnormTextureFromData(const wgpu::Device& device,
                                             size_t width,
                                             size_t height,
                                             const void* data) {
  auto w = uint32_t(width);
  auto h = uint32_t(height);
  uint32_t size = w * h;

  wgpu::TextureDescriptor desc;
  desc.size.width = w;
  desc.size.height = h;
  //desc.format = wgpu::TextureFormat::RGBA8Unorm;
  desc.format = wgpu::TextureFormat::RGBA8Unorm;
  desc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;

  auto texture = device.CreateTexture(&desc);

  wgpu::Buffer stagingBuffer = createBufferFromData(
      device, "staging buffer", data, size * 4 * sizeof(unsigned char), wgpu::BufferUsage::CopySrc);

  wgpu::ImageCopyBuffer imageCopyBuffer =
      createImageCopyBuffer(stagingBuffer, 0, w * 4 * sizeof(unsigned char), h);

  wgpu::ImageCopyTexture imageCopyTexture = createImageCopyTexture(texture, 0, {0, 0, 0});

  wgpu::Extent3D copySize = {w, h, 1};

  wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
  encoder.CopyBufferToTexture(&imageCopyBuffer, &imageCopyTexture, &copySize);
  wgpu::CommandBuffer copy = encoder.Finish();
  device.GetQueue().Submit(1, &copy);

  return texture;
}

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(
    uint32_t entryBinding,
    wgpu::ShaderStage entryVisibility,
    wgpu::BufferBindingType bufferType,
    bool bufferHasDynamicOffset,
    uint64_t bufferMinBindingSize) {
  binding = entryBinding;
  visibility = entryVisibility;
  buffer.type = bufferType;
  buffer.hasDynamicOffset = bufferHasDynamicOffset;
  buffer.minBindingSize = bufferMinBindingSize;
}

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(
    uint32_t entryBinding,
    wgpu::ShaderStage entryVisibility,
    wgpu::SamplerBindingType samplerType) {
  binding = entryBinding;
  visibility = entryVisibility;
  sampler.type = samplerType;
}

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(
    uint32_t entryBinding,
    wgpu::ShaderStage entryVisibility,
    wgpu::TextureSampleType textureSampleType,
    wgpu::TextureViewDimension textureViewDimension,
    bool textureMultisampled) {
  binding = entryBinding;
  visibility = entryVisibility;
  texture.sampleType = textureSampleType;
  texture.viewDimension = textureViewDimension;
  texture.multisampled = textureMultisampled;
}

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(
    uint32_t entryBinding,
    wgpu::ShaderStage entryVisibility,
    wgpu::StorageTextureAccess storageTextureAccess,
    wgpu::TextureFormat format,
    wgpu::TextureViewDimension textureViewDimension) {
  binding = entryBinding;
  visibility = entryVisibility;
  storageTexture.access = storageTextureAccess;
  storageTexture.format = format;
  storageTexture.viewDimension = textureViewDimension;
}

// ExternalTextureBindingLayout never contains data, so just make one that can be reused instead
// of declaring a new one every time it's needed.
// static wgpu::ExternalTextureBindingLayout kExternalTextureBindingLayout = {};

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(
    uint32_t entryBinding,
    wgpu::ShaderStage entryVisibility,
    wgpu::ExternalTextureBindingLayout* bindingLayout) {
  binding = entryBinding;
  visibility = entryVisibility;
  nextInChain = bindingLayout;
}

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(
    const wgpu::BindGroupLayoutEntry& entry)
    : wgpu::BindGroupLayoutEntry(entry) {}

wgpu::BindGroupLayout makeBindGroupLayout(
    const wgpu::Device& device,
    std::initializer_list<BindingLayoutEntryInitializationHelper> entriesInitializer) {
  std::vector<wgpu::BindGroupLayoutEntry> entries;
  for (const BindingLayoutEntryInitializationHelper& entry : entriesInitializer) {
    entries.push_back(entry);
  }

  wgpu::BindGroupLayoutDescriptor descriptor;
  descriptor.entryCount = static_cast<uint32_t>(entries.size());
  descriptor.entries = entries.data();
  return device.CreateBindGroupLayout(&descriptor);
}

BindingInitializationHelper::BindingInitializationHelper(uint32_t binding,
                                                         const wgpu::Sampler& sampler)
    : binding(binding), sampler(sampler) {}

BindingInitializationHelper::BindingInitializationHelper(uint32_t binding,
                                                         const wgpu::TextureView& textureView)
    : binding(binding), textureView(textureView) {}

BindingInitializationHelper::BindingInitializationHelper(
    uint32_t binding,
    const wgpu::ExternalTexture& externalTexture)
    : binding(binding) {
  externalTextureBindingEntry.externalTexture = externalTexture;
}

BindingInitializationHelper::BindingInitializationHelper(uint32_t binding,
                                                         const wgpu::Buffer& buffer,
                                                         uint64_t offset,
                                                         uint64_t size)
    : binding(binding), buffer(buffer), offset(offset), size(size) {}

wgpu::BindGroupEntry BindingInitializationHelper::GetAsBinding() const {
  wgpu::BindGroupEntry result;

  result.binding = binding;
  result.sampler = sampler;
  result.textureView = textureView;
  result.buffer = buffer;
  result.offset = offset;
  result.size = size;
  if (externalTextureBindingEntry.externalTexture != nullptr) {
    result.nextInChain = &externalTextureBindingEntry;
  }

  return result;
}

wgpu::BindGroup makeBindGroup(
    const wgpu::Device& device,
    const wgpu::BindGroupLayout& layout,
    std::initializer_list<BindingInitializationHelper> entriesInitializer) {
  std::vector<wgpu::BindGroupEntry> entries;
  for (const BindingInitializationHelper& helper : entriesInitializer) {
    entries.push_back(helper.GetAsBinding());
  }

  wgpu::BindGroupDescriptor descriptor;
  descriptor.layout = layout;
  descriptor.entryCount = static_cast<uint32_t>(entries.size());
  descriptor.entries = entries.data();

  return device.CreateBindGroup(&descriptor);
}

}  // namespace oak
