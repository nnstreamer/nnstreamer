/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   tensor_query_hybrid.h
 * @date   12 Oct 2021
 * @brief  Utility functions for tensor-query hybrid feature
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __TENSOR_QUERY_HYBRID_H__
#define __TENSOR_QUERY_HYBRID_H__

#include <glib.h>
#include "tensor_query_server.h"

#define DEFAULT_BROKER_HOST "tcp://localhost"
#define DEFAULT_BROKER_PORT 1883

G_BEGIN_DECLS

/**
 * @brief Data structure for node info.
 */
typedef struct
{
  gchar *host;
  guint16 port;
  query_server_info_handle server_info_h;
} query_node_info_s;

/**
 * @brief Data structure for server info.
 */
typedef struct
{
  query_node_info_s src;
  query_node_info_s sink;
} query_server_info_s;

/**
 * @brief Data structure for query-hybrid feature.
 */
typedef struct
{
  query_node_info_s node;
  query_node_info_s broker;
  gboolean is_server;
  GAsyncQueue *server_list;
  gpointer handle;
  gint state;
  gchar *topic;
} query_hybrid_info_s;

#if defined(ENABLE_QUERY_HYBRID)
/**
 * @brief Initialize query-hybrid info.
 */
extern void
tensor_query_hybrid_init (query_hybrid_info_s * info, const gchar * host, const guint16 port, gboolean is_server);

/**
 * @brief Close connection and free query-hybrid info.
 */
extern void
tensor_query_hybrid_close (query_hybrid_info_s * info);

/**
 * @brief Set current node info.
 */
extern void
tensor_query_hybrid_set_node (query_hybrid_info_s * info, const gchar * host, const guint16 port, query_server_info_handle server_info_h);

/**
 * @brief Set broker info.
 */
extern void
tensor_query_hybrid_set_broker (query_hybrid_info_s * info, const gchar * host, const guint16 port);

/**
 * @brief Get server info from node list.
 * @return Server node info. Caller should release server info using tensor_query_hybrid_free_server_info().
 */
extern query_server_info_s *
tensor_query_hybrid_get_server_info (query_hybrid_info_s * info);

/**
 * @brief Free server info.
 */
extern void
tensor_query_hybrid_free_server_info (query_server_info_s * server);

/**
 * @brief Connect to broker and publish topic.
 * @note Before calling this function, user should set broker info using tensor_query_hybrid_set_broker().
 */
extern gboolean
tensor_query_hybrid_publish (query_hybrid_info_s * info, const gchar * operation);

/**
 * @brief Connect to broker and subcribe topic.
 * @note Before calling this function, user should set broker info using tensor_query_hybrid_set_broker().
 */
extern gboolean
tensor_query_hybrid_subscribe (query_hybrid_info_s * info, const gchar * operation);
#else
#define tensor_query_hybrid_init(...)
#define tensor_query_hybrid_close(...)
#define tensor_query_hybrid_set_node(...)
#define tensor_query_hybrid_set_broker(...)
#define tensor_query_hybrid_get_server_info(...) NULL
#define tensor_query_hybrid_free_server_info(...)
#define tensor_query_hybrid_publish(...) FALSE
#define tensor_query_hybrid_subscribe(...) FALSE
#endif /* ENABLE_QUERY_HYBRID */

G_END_DECLS
#endif /* __TENSOR_QUERY_HYBRID_H__ */
