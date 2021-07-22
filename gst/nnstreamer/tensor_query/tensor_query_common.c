/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   tensor_query_common.c
 * @date   09 July 2021
 * @brief  Utility functions for tensor query
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gio/gio.h>
#include <gio/gsocket.h>
#include <stdint.h>
#include "tensor_query_common.h"

/**
 * @brief hashtable for managing server and client data
 * key: id, value: server or client data
 */
static GHashTable *table = NULL;

/** @brief Protects table */
G_LOCK_DEFINE_STATIC (splock);

/**
 * @brief Structures for tensor query server data.
 */
typedef struct
{
  TensorQueryProtocol protocol;
  GstTensorsInfo src_info;
  GstTensorsInfo sink_info;

  unsigned long long current_client_id;
  union
  {
    struct
    {
      GSocket *src_socket;
      GSocket *sink_socket;
      GCancellable *cancellable;
    };
  };
} TensorQueryServerData;

/**
 * @brief Structures for tensor query client data.
 */
typedef struct
{
  TensorQueryProtocol protocol;
  union
  {
    struct
    {
      GSocket *socket;
      GCancellable *cancellable;
    };
  };
} TensorQueryClientData;

/**
 * @brief Connect to the specified address.
 */
int
nnstreamer_query_connect (uint64_t id, const char *ip, uint32_t port,
    uint32_t timeout_ms)
{
  /** NYI: To avoid `defined but not used` */
  if (NULL == table) {
    G_LOCK (splock);
    table = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_free);
    g_hash_table_unref (table);
    G_UNLOCK (splock);
  }

  return 0;
}
