#pragma once

#include <Math/Vec.h>

#include <limits>

namespace oak {
template <class T>
bool isEqual(T a, T b, T eps = std::numeric_limits<T>::epsilon()) {
  return std::abs(a - b) < eps;
}

template <class T, size_t Dim>
bool isEqual(Vec<T, Dim> a,
             Vec<T, Dim> b,
             T eps = std::numeric_limits<T>::epsilon()) {
  for (size_t i = 0; i < Dim; ++i) {
    if (!isEqual(a[i], b[i], eps)) {
      return false;
    }
  }
  return true;
}
}  // namespace oak
