cmake_minimum_required(VERSION 3.10)

project(oak)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
include(cmake/Analyzers.cmake)

find_package(fmt)
find_package(doctest)
find_package(glfw3)
find_package(Eigen3)

add_subdirectory(third_party)

add_subdirectory(src)

