#pragma once

#include <stdio.h>
#include <stdlib.h>

#define ASSERT(expr)                                                                      \
  do {                                                                                    \
    if (!(expr)) {                                                                        \
      printf("%s:%i: Assert failed: '%s' in Function '%s'.\n", __FILE__, __LINE__, #expr, \
             __PRETTY_FUNCTION__);                                                        \
      exit(42);                                                                           \
    }                                                                                     \
  } while (false)

#define NOT_IMPLEMENTED OAK_ASSERT(!"not implemented.")
#define UNREACHABLE OAK_ASSERT(!"Unreachable code path")
