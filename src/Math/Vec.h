#pragma once

#include <algorithm>
#include <cmath>

namespace oak {

template <class T, size_t Dim>
struct VecData {
  T data[Dim];

  [[nodiscard]] T& operator[](size_t i) { return data[i]; }
  [[nodiscard]] const T& operator[](size_t i) const { return data[i]; }
};

template <class T>
struct VecData<T, 2> {
  T x, y;

  [[nodiscard]] T& operator[](size_t i) { return (&x)[i]; }
  [[nodiscard]] const T& operator[](size_t i) const { return (&x)[i]; }
};

template <class T>
struct VecData<T, 3> {
  T x, y, z;

  [[nodiscard]] T& operator[](size_t i) { return (&x)[i]; }
  [[nodiscard]] const T& operator[](size_t i) const { return (&x)[i]; }
};

template <class T>
struct VecData<T, 4> {
  T x, y, z, w;

  [[nodiscard]] T& operator[](size_t i) { return (&x)[i]; }
  [[nodiscard]] const T& operator[](size_t i) const { return (&x)[i]; }
};

template <class T, size_t Dim>
struct Vec : VecData<T, Dim> {
  Vec() {
    for (size_t i = 0; i < Dim; ++i) {
      (*this)[i] = T{0};
    }
  }

  explicit Vec(T v) {
    for (size_t i = 0; i < Dim; ++i) {
      (*this)[i] = v;
    }
  }

  template <class T1>
  explicit Vec(const Vec<T1, Dim>& other) {
    for (size_t i = 0; i < Dim; ++i) {
      (*this)[i] = T(other[i]);
    }
  }

  template <int U = Dim, std::enable_if_t<U == 4, int> = 0>
  Vec(T _x, T _y, T _z, T _w) {
    this->x = _x;
    this->y = _y;
    this->z = _z;
    this->w = _w;
  }

  template <int U = Dim, std::enable_if_t<U == 3, int> = 0>
  Vec(T _x, T _y, T _z) {
    this->x = _x;
    this->y = _y;
    this->z = _z;
  }

  template <int U = Dim, std::enable_if_t<U == 2, int> = 0>
  Vec(T _x, T _y) {
    this->x = _x;
    this->y = _y;
  }

  Vec<T, Dim>& operator+=(Vec<T, Dim> other) {
    for (size_t i = 0; i < Dim; ++i) {
      (*this)[i] += other[i];
    }
    return *this;
  }

  Vec<T, Dim>& operator-=(Vec<T, Dim> other) {
    for (size_t i = 0; i < Dim; ++i) {
      (*this)[i] -= other[i];
    }
    return *this;
  }

  Vec<T, Dim>& operator*=(Vec<T, Dim> other) {
    for (size_t i = 0; i < Dim; ++i) {
      (*this)[i] *= other[i];
    }
    return *this;
  }

  Vec<T, Dim>& operator/=(Vec<T, Dim> other) {
    for (size_t i = 0; i < Dim; ++i) {
      (*this)[i] /= other[i];
    }
    return *this;
  }

  Vec<T, Dim>& operator*=(T s) {
    for (size_t i = 0; i < Dim; ++i) {
      (*this)[i] *= s;
    }
    return *this;
  }

  Vec<T, Dim>& operator/=(T s) {
    *this *= 1. / s;
    return *this;
  }

  Vec<T, Dim> operator*(T s) {
    Vec<T, Dim> result = *this;
    result *= s;
    return result;
  }

  Vec<T, Dim> operator/(T s) {
    Vec<T, Dim> result = *this;
    result /= s;
    return result;
  }

  [[nodiscard]] T norm2() const {
    T result{0};
    for (size_t i = 0; i < Dim; ++i) {
      result += (*this)[i] * (*this)[i];
    }
    return result;
  }

  [[nodiscard]] T norm() const { return sqrt(norm2()); }
};

template <class T, size_t Dim>
Vec<T, Dim> operator*(T s, Vec<T, Dim> a) {
  return a * s;
}

template <class T, size_t Dim>
Vec<T, Dim> operator+(Vec<T, Dim> a, Vec<T, Dim> b) {
  a += b;
  return a;
}

template <class T, size_t Dim>
Vec<T, Dim> operator-(Vec<T, Dim> a, Vec<T, Dim> b) {
  a -= b;
  return a;
}

template <class T, size_t Dim>
Vec<T, Dim> operator-(Vec<T, Dim> a) {
  return Vec<T, Dim>(0) - a;
}

template <class T, size_t Dim>
Vec<T, Dim> operator*(Vec<T, Dim> a, Vec<T, Dim> b) {
  a *= b;
  return a;
}

template <class T, size_t Dim>
Vec<T, Dim> operator/(Vec<T, Dim> a, Vec<T, Dim> b) {
  a /= b;
  return a;
}

template <class T, size_t Dim>
T dot(Vec<T, Dim> a, Vec<T, Dim> b) {
  T result{0};
  for (size_t i = 0; i < Dim; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

template <class T, size_t Dim>
Vec<T, Dim> clamp(Vec<T, Dim> p, Vec<T, Dim> lower, Vec<T, Dim> upper) {
  Vec<T, Dim> result;
  for (size_t i = 0; i < Dim; ++i) {
    result[i] = std::max(std::min(p[i], upper[i]), lower[i]);
  }
  return result;
}

template <class T, size_t Dim>
Vec<T, Dim> normalize(Vec<T, Dim> v) {
  return (T(1.) / v.norm()) * v;
}

template <class T, size_t Dim>
T length(Vec<T, Dim> v) {
  return v.norm();
}

template <class T, size_t Dim>
T length2(Vec<T, Dim> v) {
  return v.norm2();
}

template <class T>
Vec<T, 3> cross(Vec<T, 3> a, Vec<T, 3> b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

using Vec2 = Vec<double, 2>;
using Vec3 = Vec<double, 3>;
using Vec4 = Vec<double, 4>;

using Vec2f = Vec<float, 2>;
using Vec3f = Vec<float, 3>;
using Vec4f = Vec<float, 4>;

using Vec2u = Vec<size_t, 2>;
using Vec3u = Vec<size_t, 3>;
using Vec4u = Vec<size_t, 4>;

}  // namespace oak
