#pragma once

#include <Core/Assert.h>

#include <span>
#include <vector>

namespace oak {

class Tensor {
 public:

  using InitListT = std::initializer_list<size_t>;

  Tensor() = default;

  explicit Tensor(std::span<const size_t> shape, double val = 0);

  Tensor(std::initializer_list<size_t> shape, double val = 0);

  void reshape(std::span<const size_t> shape);

  void reshape(std::initializer_list<size_t> shape);

  static Tensor from_file(const char* path, InitListT shape);

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

  double norm_squared() const;

 private:
  size_t m_size = 1;
  std::vector<size_t> m_shape;
  std::vector<size_t> m_strides;
  std::vector<double> m_data;
};

}  // namespace oak
