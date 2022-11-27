#pragma once

#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER // Visual Studio
#define OAK_PRETTY_FUNCTION __FUNCSIG__
#else
#define OAK_PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

#define ASSERT(expr)                                                                      \
  do {                                                                                    \
    if (!(expr)) {                                                                        \
      printf("%s:%i: Assert failed: '%s' in Function '%s'.\n", __FILE__, __LINE__, #expr, \
             OAK_PRETTY_FUNCTION);                                                        \
      exit(42);                                                                           \
    }                                                                                     \
  } while (false)

#define NOT_IMPLEMENTED ASSERT(!"not implemented.")
#define UNREACHABLE ASSERT(!"Unreachable code path")
