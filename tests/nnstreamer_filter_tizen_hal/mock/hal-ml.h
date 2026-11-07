/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file	hal-ml.h
 * @date	05 November 2026
 * @brief	Mock of the Tizen HAL ML API, enough to build the Tizen HAL sub-plugins
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs.
 */

#ifndef __MOCK_HAL_ML_H__
#define __MOCK_HAL_ML_H__

#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *hal_ml_h; /**< handle of a HAL backend */
typedef void *hal_ml_param_h; /**< handle of the parameters of a request */

/**
 * @brief Error values of the HAL ML API. Only NONE is fixed; the others are
 *        placeholders, so a test must not depend on their numbers.
 */
typedef enum {
  HAL_ML_ERROR_NONE = 0,
  HAL_ML_ERROR_INVALID_PARAMETER = -EINVAL,
  HAL_ML_ERROR_NOT_SUPPORTED = -ENOTSUP,
  HAL_ML_ERROR_RUNTIME_ERROR = -EIO,
} hal_ml_error_e;

/** @brief Create a handle of the given backend. */
int hal_ml_create (const char *backend_name, hal_ml_h *handle);

/** @brief Destroy a backend handle. */
int hal_ml_destroy (hal_ml_h handle);

/** @brief Create the parameters of a request. */
int hal_ml_param_create (hal_ml_param_h *param);

/** @brief Destroy the parameters of a request. */
int hal_ml_param_destroy (hal_ml_param_h param);

/** @brief Set a parameter of a request. */
int hal_ml_param_set (hal_ml_param_h param, const char *key, void *value);

/** @brief Send a request to the backend. */
int hal_ml_request (hal_ml_h handle, const char *request_name, hal_ml_param_h param);

/** @brief Invoke the model of the backend. */
int hal_ml_request_invoke (hal_ml_h handle, const void *input, void *output);

#ifdef __cplusplus
}
#endif

#endif /* __MOCK_HAL_ML_H__ */
