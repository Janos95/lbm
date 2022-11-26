#pragma once

#include <stdio.h>
#include <stdlib.h>

#if !defined(__PRETTY_FUNCTION__)
#define OAK_PRETTY_FUNCTION __FUNCSIG__
#elif
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
