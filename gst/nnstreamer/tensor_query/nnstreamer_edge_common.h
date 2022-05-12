/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nnstreamer_edge_common.h
 * @date   6 April 2022
 * @brief  Common util functions for nnstreamer edge.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This file is internal header for nnstreamer edge utils. DO NOT export this file.
 */

#ifndef __NNSTREAMER_EDGE_COMMON_H__
#define __NNSTREAMER_EDGE_COMMON_H__

#include <glib.h> /** @todo remove glib */
#include "nnstreamer_edge.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Utility to silence unused parameter warning for intentionally unused parameters (e.g., callback functions of a framework)
 */
#ifndef UNUSED
#define UNUSED(expr) do { (void)(expr); } while (0)
#endif

/**
 * @brief g_memdup() function replaced by g_memdup2() in glib version >= 2.68
 */
#if GLIB_USE_G_MEMDUP2
#define _g_memdup g_memdup2
#else
#define _g_memdup g_memdup
#endif

#define NNS_EDGE_MAGIC 0xfeedfeed
#define NNS_EDGE_MAGIC_DEAD 0xdeaddead
#define NNS_EDGE_MAGIC_IS_VALID(h) ((h) && (h)->magic == NNS_EDGE_MAGIC)

/**
 * @brief Internal data structure for raw data.
 */
typedef struct {
  void *data;
  size_t data_len;
  nns_edge_data_destroy_cb destroy_cb;
} nns_edge_raw_data_s;

/**
 * @brief Internal data structure for edge data.
 * @todo Implement mutex lock.
 */
typedef struct {
  unsigned int magic;
  nns_edge_data_type_e dtype;
  unsigned int num;
  nns_edge_raw_data_s data[NNS_EDGE_DATA_LIMIT];
} nns_edge_data_s;

/**
 * @brief Internal data structure for edge event.
 * @todo Implement mutex lock.
 */
typedef struct {
  unsigned int magic;
  nns_edge_event_e event;
  nns_edge_raw_data_s data;
} nns_edge_event_s;

/**
 * @todo add log util for nnstreamer-edge.
 * 1. define tag (e.g., "nnstreamer-edge").
 * 2. consider macros to print function and line.
 * 3. new API to get last error.
 */
#define nns_edge_logi g_info
#define nns_edge_logw g_warning
#define nns_edge_loge g_critical
#define nns_edge_logd g_debug
#define nns_edge_logf g_error

/**
 * @brief Validate edge handle.
 */
bool nns_edge_handle_is_valid (nns_edge_h edge_h);

/**
 * @brief Check network connection.
 */
bool nns_edge_is_connected (nns_edge_h edge_h);

/**
 * @brief Validate data handle.
 */
bool nns_edge_data_is_valid (nns_edge_data_h data_h);

/**
 * @brief Create nnstreamer edge event.
 */
int nns_edge_event_create (nns_edge_event_e event, nns_edge_event_h * event_h);

/**
 * @brief Destroy nnstreamer edge event.
 */
int nns_edge_event_destroy (nns_edge_event_h event_h);

/**
 * @brief Set event data.
 */
int nns_edge_event_set_data (nns_edge_event_h event_h, void *data, size_t data_len, nns_edge_data_destroy_cb destroy_cb);

/**
 * @brief Get the nnstreamer edge event type.
 */
int nns_edge_event_get_type (nns_edge_event_h event_h, nns_edge_event_e * event);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __NNSTREAMER_EDGE_COMMON_H__ */
