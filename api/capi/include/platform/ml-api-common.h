/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 */
/**
 * @file ml-api-common.h
 * @date 24 October 2019
 * @brief ml-api-common header for Non-Tizen platforms
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author Wook Song <wook16.song@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __platform_ml_api_common_H__
#define __platform_ml_api_common_H__

#ifdef __TIZEN__

#error "Tizen platform should not use this header"

#endif

#include <errno.h>

/**
 @ref: https://gitlab.freedesktop.org/dude/gst-plugins-base/commit/89095e7f91cfbfe625ec2522da49053f1f98baf8
 */
#if !defined(ESTRPIPE)
#define ESTRPIPE EPIPE
#endif /* !defined(ESTRPIPE) */

/**
 * @brief Enumeration for the error codes of NNStreamer for None-tizen platform
 */
typedef enum {
  ML_ERROR_NONE                 = 0, /**< Success! */
  ML_ERROR_INVALID_PARAMETER    = -EINVAL, /**< Invalid parameter */
  ML_ERROR_STREAMS_PIPE         = -ESTRPIPE, /**< Cannot create or access the pipeline. */
  ML_ERROR_TRY_AGAIN            = -EAGAIN, /**< The pipeline is not ready, yet (not negotiated, yet) */
  ML_ERROR_UNKNOWN              = -0x40000000LL,  /**< Unknown error */
  ML_ERROR_TIMED_OUT            = -ML_ERROR_UNKNOWN + 1,  /**< Time out */
  ML_ERROR_NOT_SUPPORTED        = -ML_ERROR_UNKNOWN + 2, /**< The feature is not supported */
  ML_ERROR_PERMISSION_DENIED    = -EACCES, /**< Permission denied */
  ML_ERROR_OUT_OF_MEMORY        = -ENOMEM, /**< Out of memory (Since 6.0) */
} ml_error_e;

#endif /* __platform_ml_api_common_H__ */
