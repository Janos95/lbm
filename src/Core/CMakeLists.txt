add_library(OakCore INTERFACE)
target_include_directories(OakCore INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(Oak::Core ALIAS OakCore)

