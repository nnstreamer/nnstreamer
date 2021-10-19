/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_meta.h
 * @date    09 Aug 2021
 * @brief   Internal tensor meta header for nnstreamer
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifndef __GST_TENSOR_META_H__
#define __GST_TENSOR_META_H__

#include <gst/gst.h>
#include <tensor_typedef.h>

G_BEGIN_DECLS

typedef int64_t query_client_id_t;

/**
 * @brief GstMetaQuery meta structure
 */
typedef struct
{
  GstMeta meta;

  query_client_id_t client_id;
} GstMetaQuery;

/**
 * @brief Define meta_query type to register
 */
GType gst_meta_query_api_get_type (void);
#define GST_META_QUERY_API_TYPE (gst_meta_query_api_get_type())

/**
 * @brief Get meta_query info
 */
const GstMetaInfo * gst_meta_query_get_info (void);
#define GST_META_QUERY_INFO (gst_meta_query_get_info())
#define gst_buffer_get_meta_query(b) \
    ((GstMetaQuery *) gst_buffer_get_meta ((b), GST_META_QUERY_API_TYPE))
#define gst_buffer_add_meta_query(b) \
    ((GstMetaQuery *) gst_buffer_add_meta ((b), GST_META_QUERY_INFO, NULL))

G_END_DECLS

#endif /* __GST_TENSOR_META_H__ */
