#include "Tensor.h"

#include <fstream>

namespace oak {

Tensor::Tensor(std::span<const size_t> shape, double val) {
  reshape(shape);
  m_data.resize(m_size, val);
}

Tensor::Tensor(std::initializer_list<size_t> shape, double val)
    : Tensor(std::span<const size_t>(shape.begin(), shape.size()), val) {}

void Tensor::reshape(std::span<const size_t> shape) {
  ASSERT(!shape.empty());
  m_shape = std::vector(shape.begin(), shape.end());
  m_strides.resize(shape.size());
  m_strides.back() = 1;
  for (int i = int(shape.size()) - 1; i >= 0; --i) {
    m_size *= shape.begin()[i];
    auto j = size_t(i) + 1;
    m_strides[size_t(i)] = j == shape.size() ? 1 : m_strides[j] * m_shape[j];
  }
}

void Tensor::reshape(std::initializer_list<size_t> shape) {
    reshape(std::span<const size_t>(shape.begin(), shape.size()));
}

Tensor Tensor::from_file(const char* path, InitListT shape) {
  std::ifstream ifs(path);
  std::string line;
  std::vector<double> values;
  while (std::getline(ifs, line)) {
    double v = std::stod(line);
    values.push_back(v);
  }
  Tensor t(shape);
  for (size_t i = 0; i < values.size(); ++i) {
        t[i] = values[i];
  }
  return t;
}

double Tensor::norm_squared() const {
  double n2 = 0.;
  for (double x : m_data) {
    n2 += x * x;
  }
  return n2;
}

}  // namespace oak