/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   tensor_query_common.c
 * @date   09 July 2021
 * @brief  Utility functions for tensor query
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @author Junhwan Kim <jejudo.kim@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "tensor_query_common.h"

#ifndef EREMOTEIO
#define EREMOTEIO 121           /* This is Linux-specific. Define this for non-Linux systems */
#endif

/**
 * @brief Register GEnumValue array for query connect-type property.
 */
GType
gst_tensor_query_get_connect_type (void)
{
  static GType protocol = 0;
  if (protocol == 0) {
    static GEnumValue protocols[] = {
      {NNS_EDGE_CONNECT_TYPE_TCP, "TCP",
          "Directly sending stream frames via TCP connections."},
      {NNS_EDGE_CONNECT_TYPE_HYBRID, "HYBRID",
          "Connect with MQTT brokers and directly sending stream frames via TCP connections."},
      {0, NULL, NULL},
    };
    protocol = g_enum_register_static ("tensor_query_protocol", protocols);
  }

  return protocol;
}

/**
 * @brief Get the tensors config to check the data from a remote peer with, from the caps of the receiving pad.
 * @param[in] caps The current caps of the receiving pad (nullable).
 * @param[out] config The tensors config. If @a caps describe tensors that cannot be parsed, it describes no tensor, so that every data is refused.
 * @return TRUE if @a caps are fixed tensor caps and the data should be checked with @a config.
 * @note The framerate is not required. The caller should free @a config with gst_tensors_config_free().
 */
gboolean
gst_tensor_query_config_from_caps (GstCaps * caps, GstTensorsConfig * config)
{
  g_return_val_if_fail (config != NULL, FALSE);

  gst_tensors_config_init (config);

  if (!caps || !gst_caps_is_fixed (caps) ||
      !gst_structure_is_tensor_stream (gst_caps_get_structure (caps, 0)))
    return FALSE;

  if (!gst_tensors_config_from_caps (config, caps, FALSE) ||
      !gst_tensors_info_validate (&config->info)) {
    gst_tensors_config_free (config);
    gst_tensors_config_init (config);
  }

  return TRUE;
}

/**
 * @brief Check the memories of edge data received from a remote peer against the negotiated tensors config.
 * @param[in] data_h The edge data received from the peer.
 * @param[in] config The tensors config of the negotiated caps.
 * @return TRUE if the memories are what @a config describes, FALSE otherwise.
 * @note For static tensors, the number of memories and the size of each memory should be same as @a config.
 *       For flexible and sparse tensors, each memory should have a valid header and the data described by the header.
 */
gboolean
gst_tensor_query_validate_edge_data (nns_edge_data_h data_h,
    GstTensorsConfig * config)
{
  GstTensorMetaInfo meta;
  GstTensorInfo *info;
  guint i, num_data;
  void *data;
  nns_size_t data_len;
  gsize hsize, expected;
  gboolean is_static;

  g_return_val_if_fail (config != NULL, FALSE);

  if (nns_edge_data_get_count (data_h, &num_data) != NNS_EDGE_ERROR_NONE)
    return FALSE;

  is_static = gst_tensors_config_is_static (config);
  if (is_static && num_data != config->info.num_tensors) {
    nns_loge
        ("The edge data has %u memories, but the caps describe %u tensors.",
        num_data, config->info.num_tensors);
    return FALSE;
  }

  if (num_data == 0 || num_data > NNS_TENSOR_SIZE_LIMIT) {
    nns_loge ("Invalid number of memories in the edge data: %u.", num_data);
    return FALSE;
  }

  for (i = 0; i < num_data; i++) {
    if (nns_edge_data_get (data_h, i, &data, &data_len) != NNS_EDGE_ERROR_NONE)
      return FALSE;

    if (is_static) {
      info = gst_tensors_info_get_nth_info (&config->info, i);
      expected = gst_tensor_info_get_size (info);

      if (data_len != expected) {
        nns_loge ("The %u-th memory of the edge data has %" G_GSIZE_FORMAT
            " bytes, but the caps describe %" G_GSIZE_FORMAT " bytes.", i,
            (gsize) data_len, expected);
        return FALSE;
      }
    } else {
      gst_tensor_meta_info_init (&meta);
      hsize = gst_tensor_meta_info_get_header_size (&meta);

      if (data_len < hsize || !gst_tensor_meta_info_parse_header (&meta, data)) {
        nns_loge ("The %u-th memory of the edge data has no valid header.", i);
        return FALSE;
      }

      hsize = gst_tensor_meta_info_get_header_size (&meta);
      expected = gst_tensor_meta_info_get_data_size (&meta);

      if (data_len < hsize || data_len - hsize < expected) {
        nns_loge ("The %u-th memory of the edge data has %" G_GSIZE_FORMAT
            " bytes, too small for its header describing %" G_GSIZE_FORMAT
            " bytes of data.", i, (gsize) data_len, expected);
        return FALSE;
      }
    }
  }

  return TRUE;
}

/**
 * @brief Push edge data received from a remote peer to the receive queue of an element.
 * @param[in] queue The receive queue.
 * @param[in] data_h The edge data to push (transfer full).
 * @param[in] max_buffers The maximum number of data in the queue, 0 for no limit.
 * @return The number of data dropped to keep the queue within @a max_buffers.
 * @note If the queue is full, the oldest data are dropped, so the queue keeps the latest data.
 */
guint
gst_tensor_query_push_edge_data (GAsyncQueue * queue, nns_edge_data_h data_h,
    guint max_buffers)
{
  nns_edge_data_h old_h;
  gint length;
  guint dropped = 0;

  g_return_val_if_fail (queue != NULL, 0);
  g_return_val_if_fail (data_h != NULL, 0);

  g_async_queue_lock (queue);
  if (max_buffers > 0) {
    while ((length = g_async_queue_length_unlocked (queue)) > 0
        && (guint) length >= max_buffers
        && (old_h = g_async_queue_try_pop_unlocked (queue)) != NULL) {
      nns_edge_data_destroy (old_h);
      dropped++;
    }
  }
  g_async_queue_push_unlocked (queue, data_h);
  g_async_queue_unlock (queue);

  return dropped;
}
