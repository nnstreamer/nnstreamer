/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_serversink.h
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query_server sink
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifndef __GST_TENSOR_QUERY_SERVERSINK_H__
#define __GST_TENSOR_QUERY_SERVERSINK_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <tensor_common.h>
#include <tensor_meta.h>
#include "tensor_query_common.h"
#include "tensor_query_server.h"

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_QUERY_SERVERSINK \
  (gst_tensor_query_serversink_get_type())
#define GST_TENSOR_QUERY_SERVERSINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_QUERY_SERVERSINK,GstTensorQueryServerSink))
#define GST_TENSOR_QUERY_SERVERSINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_QUERY_SERVERSINK,GstTensorQueryServerSinkClass))
#define GST_IS_TENSOR_QUERY_SERVERSINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_QUERY_SERVERSINK))
#define GST_IS_TENSOR_QUERY_SERVERSINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_QUERY_SERVERSINK))
#define GST_TENSOR_QUERY_SERVERSINK_CAST(obj) ((GstTensorQueryServerSink *)(obj))

typedef struct _GstTensorQueryServerSink GstTensorQueryServerSink;
typedef struct _GstTensorQueryServerSinkClass GstTensorQueryServerSinkClass;

/**
 * @brief GstTensorQueryServerSink data structure.
 */
struct _GstTensorQueryServerSink
{
  GstBaseSink element; /**< parent object */
  guint sink_id;

  guint16 port;
  gchar *host;
  TensorQueryProtocol protocol;
  guint timeout;
  query_server_handle server_data;
  query_server_info_handle server_info_h;
  gint metaless_frame_limit;
  gint metaless_frame_count;
};

/**
 * @brief GstTensorQueryServerSinkClass data structure.
 */
struct _GstTensorQueryServerSinkClass
{
  GstBaseSinkClass parent_class; /**< parent class */
};

GType gst_tensor_query_serversink_get_type (void);

G_END_DECLS
#endif /* __GST_TENSOR_QUERY_SERVERSINK_H__ */
