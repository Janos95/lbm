
add_executable(OakApp
        main.cpp
        Application.cpp
        Application.h
        callbacks.cc
        callbacks.h
        dump_utils.cc
        dump_utils.h
        WebGpuHelpers.cpp
        WebGpuHelpers.h
        wgpu.h
        ColorMaps.h)

target_link_libraries(OakApp
        Oak::Math
        fmt::fmt
        TBB::tbb
        webgpu_dawn
        webgpu_cpp
        webgpu_glfw
        glfw
        )

#add_warnings(OakApp)
target_include_directories(OakApp PRIVATE "${PROJECT_SOURCE_DIR}/third_party/dawn/include")
