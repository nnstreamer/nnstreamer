/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_client.h
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query client
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifndef __GST_TENSOR_QUERY_CLIENT_H__
#define __GST_TENSOR_QUERY_CLIENT_H__

#include <gst/gst.h>
#include <gio/gio.h>
#include <tensor_common.h>
#include "tensor_query_common.h"
#include "tensor_query_hybrid.h"

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_QUERY_CLIENT \
  (gst_tensor_query_client_get_type())
#define GST_TENSOR_QUERY_CLIENT(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_QUERY_CLIENT,GstTensorQueryClient))
#define GST_TENSOR_QUERY_CLIENT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_QUERY_CLIENT,GstTensorQueryClientClass))
#define GST_IS_TENSOR_QUERY_CLIENT(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_QUERY_CLIENT))
#define GST_IS_TENSOR_QUERY_CLIENT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_QUERY_CLIENT))
#define GST_TENSOR_QUERY_CLIENT_CAST(obj) ((GstTensorQueryClient *)(obj))

typedef struct _GstTensorQueryClient GstTensorQueryClient;
typedef struct _GstTensorQueryClientClass GstTensorQueryClientClass;

/**
 * @brief GstTensorQueryClient data structure.
 */
struct _GstTensorQueryClient
{
  GstElement element; /**< parent object */
  GstPad *sinkpad; /**< sink pad */
  GstPad *srcpad; /**< src pad */

  gboolean silent; /**< True if logging is minimized */
  gchar *in_caps_str;

  TensorQueryProtocol protocol;

  /* Query-hybrid feature */
  gchar *operation; /**< Main operation such as 'object_detection' or 'image_segmentation' */
  query_hybrid_info_s hybrid_info;
  gchar *broker_host;
  guint16 broker_port;

  /* src information (Connect to query server source) */
  query_connection_handle src_conn;
  gchar *src_host;
  guint16 src_port;

  /* sink socket and information (Connect to query server sink) */
  query_connection_handle sink_conn;
  gchar *sink_host;
  guint16 sink_port;
};

/**
 * @brief GstTensorQueryClientClass data structure.
 */
struct _GstTensorQueryClientClass
{
  GstElementClass parent_class; /**< parent class */
};

GType gst_tensor_query_client_get_type (void);

G_END_DECLS
#endif /* __GST_TENSOR_QUERY_CLIENT_H__ */
