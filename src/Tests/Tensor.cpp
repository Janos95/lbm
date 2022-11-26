#include <doctest/doctest.h>

#include <Tensor.h>
#include <Utility.h>

#include <array>

using namespace oak;

#define CHECK_FUZZY(a, b) CHECK(isEqual((a), (b)))

TEST_CASE("Tensor") {
  std::array<size_t, 3> shape = {2, 3, 4};
  Tensor t1(shape, 1.);
  Tensor t2({2, 3, 4}, 1.);
  Tensor t3({2, 3, 4});
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; i < shape[1]; ++j) {
      for (size_t k = 0; k < shape[2]; ++k) {
        t3(i, j, k) = t1(i, j, k) + t2(i, j, k);
      }
    }
  }

  double sum = 0;

  for (size_t i = 0; i < t3.size(); ++i) {
    sum += t3[i];
  }

  CHECK_FUZZY(sum, 2. * 3. * 4. * 2.);
}
