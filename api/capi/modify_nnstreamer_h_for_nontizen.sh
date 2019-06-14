#!/usr/bin/env bash

sed -i "s|#include <tizen_error.h>|#include <errno.h>\n\
#define TIZEN_ERROR_NONE (0)\n\
#define TIZEN_ERROR_INVALID_PARAMETER (-EINVAL)\n\
#define TIZEN_ERROR_STREAMS_PIPE (-ESTRPIPE)\n\
#define TIZEN_ERROR_TRY_AGAIN (-EAGAIN)\n\
#define TIZEN_ERROR_UNKNOWN (-1073741824LL)\n\
#define TIZEN_ERROR_TIMED_OUT (TIZEN_ERROR_UNKNOWN + 1)\n\
#define TIZEN_ERROR_NOT_SUPPORTED (TIZEN_ERROR_UNKNOWN + 2)\
|" include/nnstreamer.h
