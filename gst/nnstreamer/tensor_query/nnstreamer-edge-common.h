/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nnstreamer-edge-common.h
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
#include <fcntl.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <pthread.h>
#include <unistd.h>
#include "nnstreamer-edge.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Utility to silence unused parameter warning for intentionally unused parameters (e.g., callback functions of a framework)
 */
#ifndef UNUSED
#define UNUSED(expr) do { (void)(expr); } while (0)
#endif

#define STR_IS_VALID(s) ((s) && (s)[0] != '\0')
#define SAFE_FREE(p) do { if (p) { free (p); (p) = NULL; } } while (0)

#define NNS_EDGE_MAGIC 0xfeedfeed
#define NNS_EDGE_MAGIC_DEAD 0xdeaddead
#define NNS_EDGE_MAGIC_IS_VALID(h) ((h) && (h)->magic == NNS_EDGE_MAGIC)

#define nns_edge_lock_init(h) do { pthread_mutex_init (&(h)->lock, NULL); } while (0)
#define nns_edge_lock_destroy(h) do { pthread_mutex_destroy (&(h)->lock); } while (0)
#define nns_edge_lock(h) do { pthread_mutex_lock (&(h)->lock); } while (0)
#define nns_edge_unlock(h) do { pthread_mutex_unlock (&(h)->lock); } while (0)

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
 */
typedef struct {
  unsigned int magic;
  unsigned int num;
  nns_edge_raw_data_s data[NNS_EDGE_DATA_LIMIT];
  GHashTable *info_table;
} nns_edge_data_s;

/**
 * @brief Internal data structure for edge event.
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
 * @brief Internal util function to get available port number.
 */
int nns_edge_get_available_port (void);

/**
 * @brief Free allocated memory.
 */
void nns_edge_free (void *data);

/**
 * @brief Allocate new memory and copy bytes.
 * @note Caller should release newly allocated memory using nns_edge_free().
 */
void *nns_edge_memdup (const void *data, size_t size);

/**
 * @brief Allocate new memory and copy string.
 * @note Caller should release newly allocated string using nns_edge_free().
 */
char *nns_edge_strdup (const char *str);

/**
 * @brief Allocate new memory and print formatted string.
 * @note Caller should release newly allocated string using nns_edge_free().
 */
char *nns_edge_strdup_printf (const char *format, ...);

/**
 * @brief Create nnstreamer edge event.
 * @note This is internal function for edge event.
 */
int nns_edge_event_create (nns_edge_event_e event, nns_edge_event_h * event_h);

/**
 * @brief Destroy nnstreamer edge event.
 * @note This is internal function for edge event.
 */
int nns_edge_event_destroy (nns_edge_event_h event_h);

/**
 * @brief Set event data.
 * @note This is internal function for edge event.
 */
int nns_edge_event_set_data (nns_edge_event_h event_h, void *data, size_t data_len, nns_edge_data_destroy_cb destroy_cb);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __NNSTREAMER_EDGE_COMMON_H__ */
