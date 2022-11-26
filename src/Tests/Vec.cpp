#include <doctest/doctest.h>

#include <Utility.h>
#include <Vec.h>

using namespace oak;

#define CHECK_FUZZY(a, b) CHECK(isEqual((a), (b)))

TEST_CASE("Vec2") {
  Vec2 v, w;
  CHECK_EQ(v.x, 0.);
  CHECK_EQ(v.y, 0.);

  v = {1., 2.};
  w = {2., -1.};
  CHECK_FUZZY(dot(v, w), 0.);

  CHECK_FUZZY(v + w, Vec2(3., 1.));
  CHECK_FUZZY(v - w, Vec2(-1., 3.));
  CHECK_FUZZY(v * w, Vec2(2., -2.));
  CHECK_FUZZY(v / w, Vec2(0.5, -2.));
  CHECK_FUZZY(v * 2., Vec2(2., 4.));
  CHECK_FUZZY(2. * w, Vec2(4., -2.));

  auto v1 = v;
  v1 += w;
  auto v2 = v;
  v2 -= w;
  auto v3 = v;
  v3 *= w;
  auto v4 = v;
  v4 /= w;
  auto v5 = v;
  v5 *= 2.;

  CHECK_FUZZY(v1, Vec2(3., 1.));
  CHECK_FUZZY(v2, Vec2(-1., 3.));
  CHECK_FUZZY(v3, Vec2(2., -2.));
  CHECK_FUZZY(v4, Vec2(0.5, -2.));
  CHECK_FUZZY(v5, Vec2(2., 4.));

  CHECK_FUZZY(v.norm(), std::sqrt(5.));
  CHECK_FUZZY(v.norm2(), 5.);
}

TEST_CASE("Vec3") {
  Vec3 v, w;
  CHECK_EQ(v.x, 0.);
  CHECK_EQ(v.y, 0.);
  CHECK_EQ(v.z, 0.);

  v = {1., 2., 3.};
  w = {2., -1., 3.};

  auto s = cross(v, w);
  CHECK_FUZZY(dot(v, s), 0.);
  CHECK_FUZZY(dot(w, s), 0.);

  CHECK_FUZZY(v + w, Vec3(3., 1., 6.));
  CHECK_FUZZY(v - w, Vec3(-1., 3., 0.));
  CHECK_FUZZY(v * w, Vec3(2., -2., 9.));
  CHECK_FUZZY(v / w, Vec3(0.5, -2., 1.));
  CHECK_FUZZY(v * 2., Vec3(2., 4., 6.));
  CHECK_FUZZY(2. * w, Vec3(4., -2., 6.));

  auto v1 = v;
  v1 += w;
  auto v2 = v;
  v2 -= w;
  auto v3 = v;
  v3 *= w;
  auto v4 = v;
  v4 /= w;
  auto v5 = v;
  v5 *= 2.;

  CHECK_FUZZY(v1, Vec3(3., 1., 6.));
  CHECK_FUZZY(v2, Vec3(-1., 3., 0.));
  CHECK_FUZZY(v3, Vec3(2., -2., 9.));
  CHECK_FUZZY(v4, Vec3(0.5, -2., 1.));
  CHECK_FUZZY(v5, Vec3(2., 4., 6.));

  CHECK_FUZZY(v.norm(), std::sqrt(14.));
  CHECK_FUZZY(v.norm2(), 14.);
}