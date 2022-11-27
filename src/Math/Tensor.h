#pragma once

#include <Core/Assert.h>
#include <span>
#include <vector>

namespace oak {

class Tensor {
 public:
  Tensor() = default;

  Tensor(std::span<const size_t> shape, double val = 0)
      : m_shape(shape.begin(), shape.end()), m_strides(shape.size()) {
    m_strides[0] = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      m_size *= shape.begin()[ptrdiff_t(i)];
      if (i > 0) {
        m_strides[i] = m_strides[i - 1] * m_shape[i - 1];
      }
    }
    m_data.resize(m_size, val);
  }

  Tensor(std::initializer_list<size_t> shape, double val = 0)
      : Tensor(std::span<const size_t>(shape.begin(), shape.size()), val) {}

  template <class... Args>
    requires((std::is_same_v<Args, size_t> && ...))
  const double& operator()(Args... indices) const {
    ASSERT(sizeof...(Args) == m_shape.size());
    size_t idx = 0;
    size_t is[] = {indices...};
    for (size_t i = 0; i < sizeof...(Args); ++i) {
      idx += is[i] * m_strides[i];
      ASSERT(is[i] < m_shape[i]);
    }
    ASSERT(idx < m_data.size());
    return m_data[idx];
  }

  template <class... Args>
    requires((std::is_same_v<Args, size_t> && ...))
  double& operator()(Args... indices) {
    ASSERT(sizeof...(Args) == m_shape.size());
    size_t idx = 0;
    size_t is[] = {indices...};
    for (size_t i = 0; i < sizeof...(Args); ++i) {
      idx += is[i] * m_strides[i];
      ASSERT(is[i] < m_shape[i]);
    }
    ASSERT(idx < m_data.size());
    return m_data[idx];
  }

  double& operator[](size_t i) { return m_data[i]; }

  const double& operator[](size_t i) const { return m_data[i]; }

  size_t size() const { return m_size; }

  std::span<const size_t> shape() const { return m_shape; }

 private:
  size_t m_size = 1;
  std::vector<size_t> m_shape;
  std::vector<size_t> m_strides;
  std::vector<double> m_data;
};

}  // namespace oak
