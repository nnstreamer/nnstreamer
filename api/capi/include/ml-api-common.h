/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer API / Tizen Machine-Learning API Common Header
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	ml-api-common.h
 * @date	07 May 2020
 * @brief	ML-API Common Header
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @details
 *      More entries might be migrated from nnstreamer.h if
 *    other modules of Tizen-ML APIs use common data structures.
 */
#ifndef __ML_API_COMMON_H__
#define __ML_API_COMMON_H__

#include <tizen_error.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Enumeration for the error codes of NNStreamer.
 * @since_tizen 5.5
 */
typedef enum {
  ML_ERROR_NONE                 = TIZEN_ERROR_NONE, /**< Success! */
  ML_ERROR_INVALID_PARAMETER    = TIZEN_ERROR_INVALID_PARAMETER, /**< Invalid parameter */
  ML_ERROR_STREAMS_PIPE         = TIZEN_ERROR_STREAMS_PIPE, /**< Cannot create or access the pipeline. */
  ML_ERROR_TRY_AGAIN            = TIZEN_ERROR_TRY_AGAIN, /**< The pipeline is not ready, yet (not negotiated, yet) */
  ML_ERROR_UNKNOWN              = TIZEN_ERROR_UNKNOWN,  /**< Unknown error */
  ML_ERROR_TIMED_OUT            = TIZEN_ERROR_TIMED_OUT,  /**< Time out */
  ML_ERROR_NOT_SUPPORTED        = TIZEN_ERROR_NOT_SUPPORTED, /**< The feature is not supported */
  ML_ERROR_PERMISSION_DENIED    = TIZEN_ERROR_PERMISSION_DENIED, /**< Permission denied */
  ML_ERROR_OUT_OF_MEMORY        = TIZEN_ERROR_OUT_OF_MEMORY, /**< Out of memory (Since 6.0) */
} ml_error_e;

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __ML_API_COMMON_H__ */
