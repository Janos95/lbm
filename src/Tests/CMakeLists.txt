add_executable(tests Vec.cpp tests.cpp Tensor.cpp)
target_link_libraries(tests PRIVATE
        doctest::doctest
        Oak::Math
        )
