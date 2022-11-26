add_library(OakMath INTERFACE)
target_include_directories(OakMath INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(OakMath INTERFACE OakCore)

add_library(Oak::Math ALIAS OakMath)
