function(add_warnings TARGET)
  target_compile_options(${TARGET} PRIVATE
          -fvisibility=hidden
          -Wall
          -Werror
          -Wextra
          -Wno-unknown-pragmas
          -Wpedantic
          -pedantic-errors
          )

  if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
  ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
    target_compile_options(${TARGET} PRIVATE
            -Weverything
            -Wno-c++98-compat
            -Wno-c++98-compat-pedantic
            -Wno-documentation
            -Wno-documentation-unknown-command
            -Wno-double-promotion
            -Wno-padded
            -Wno-poison-system-directories
            -Wno-shadow-field-in-constructor
            -Wno-zero-as-null-pointer-constant
            )
  elseif(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET} PRIVATE
            -Wno-missing-field-initializers
            )
  endif()
endfunction()