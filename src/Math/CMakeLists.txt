add_library(OakMath Tensor.h Tensor.cpp Utility.h Vec.h)

target_include_directories(OakMath PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(OakMath PUBLIC OakCore)

add_library(Oak::Math ALIAS OakMath)
