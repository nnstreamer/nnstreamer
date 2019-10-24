/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 */
/**
 * @file tizen_error.h
 * @date 24 October 2019
 * @brief C-API internal header emulating tizen_error.h for Non-Tizen platforms
 * @see	https://github.com/nnsuite/nnstreamer
 * @author Wook Song <wook16.song@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __NTZN_TIZEN_ERROR_H__
#define __NTZN_TIZEN_ERROR_H__

#include <errno.h>
#define TIZEN_ERROR_NONE (0)
#define TIZEN_ERROR_INVALID_PARAMETER (-EINVAL)
#define TIZEN_ERROR_STREAMS_PIPE (-ESTRPIPE)
#define TIZEN_ERROR_TRY_AGAIN (-EAGAIN)
#define TIZEN_ERROR_UNKNOWN (-1073741824LL)
#define TIZEN_ERROR_TIMED_OUT (TIZEN_ERROR_UNKNOWN + 1)
#define TIZEN_ERROR_NOT_SUPPORTED (TIZEN_ERROR_UNKNOWN + 2)
#define TIZEN_ERROR_PERMISSION_DENIED (-EACCES)

#endif /* __NTZN_TIZEN_ERROR_H__ */
