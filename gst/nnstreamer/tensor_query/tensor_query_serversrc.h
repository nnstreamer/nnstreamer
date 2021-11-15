/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_serversrc.h
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query_server src
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifndef __GST_TENSOR_QUERY_SERVERSRC_H__
#define __GST_TENSOR_QUERY_SERVERSRC_H__

#include <gst/base/gstbasesrc.h>
#include <gst/base/gstpushsrc.h>
#include "tensor_query_common.h"
#include <nnstreamer_util.h>
#include <tensor_meta.h>
#include "tensor_query_hybrid.h"

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_QUERY_SERVERSRC \
  (gst_tensor_query_serversrc_get_type())
#define GST_TENSOR_QUERY_SERVERSRC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_QUERY_SERVERSRC,GstTensorQueryServerSrc))
#define GST_TENSOR_QUERY_SERVERSRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_QUERY_SERVERSRC,GstTensorQueryServerSrcClass))
#define GST_IS_TENSOR_QUERY_SERVERSRC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_QUERY_SERVERSRC))
#define GST_IS_TENSOR_QUERY_SERVERSRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_QUERY_SERVERSRC))
#define GST_TENSOR_QUERY_SERVERSRC_CAST(obj) ((GstTensorQueryServerSrc *)(obj))

typedef struct _GstTensorQueryServerSrc GstTensorQueryServerSrc;
typedef struct _GstTensorQueryServerSrcClass GstTensorQueryServerSrcClass;

/**
 * @brief GstTensorQueryServerSrc data structure.
 */
struct _GstTensorQueryServerSrc
{
  GstPushSrc element; /* parent object */
  guint src_id;
  gboolean configured;

  guint16 port;
  gchar *host;
  TensorQueryProtocol protocol;
  guint timeout;

  /* Query-hybrid feature */
  gchar *operation; /**< Main operation such as 'object_detection' or 'image_segmentation' */
  query_hybrid_info_s hybrid_info;
  gchar *broker_host;
  guint16 broker_port;

  query_server_handle server_data; /* server data passed to common functions */
  query_server_info_handle server_info_h;
};

/**
 * @brief GstTensorQueryServerSrcClass data structure.
 */
struct _GstTensorQueryServerSrcClass
{
  GstPushSrcClass parent_class; /**< parent class */
};

GType gst_tensor_query_serversrc_get_type (void);

G_END_DECLS
#endif /* __GST_TENSOR_QUERY_SERVERSRC_H__ */
