/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Src_gRPC
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    tensor_src_grpc.h
 * @date    20 Oct 2020
 * @brief   GStreamer plugin to support gRPC tensor source
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_SRC_GRPC_H__
#define __GST_TENSOR_SRC_GRPC_H__

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <gst/base/gstdataqueue.h>

#include <nnstreamer_protobuf_grpc.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_SRC_GRPC \
  (gst_tensor_src_grpc_get_type())
#define GST_TENSOR_SRC_GRPC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_SRC_GRPC,GstTensorSrcGRPC))
#define GST_TENSOR_SRC_GRPC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_SRC_GRPC,GstTensorSrcGRPCClass))
#define GST_IS_TENSOR_SRC_GRPC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_SRC_GRPC))
#define GST_IS_TENSOR_SRC_GRPC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_SRC_GRPC))
#define GST_TENSOR_SRC_GRPC_CAST(obj)  ((GstTensorSrcGRPC *)(obj))
typedef struct _GstTensorSrcGRPC GstTensorSrcGRPC;
typedef struct _GstTensorSrcGRPCClass GstTensorSrcGRPCClass;

typedef enum
{
  GST_TENSOR_SRC_GRPC_CONFIGURED = (GST_ELEMENT_FLAG_LAST << 0),
  GST_TENSOR_SRC_GRPC_STARTED = (GST_ELEMENT_FLAG_LAST << 1),
} GstTensorSrcGRPCFlags;

/**
 * @brief GstTensorSrcGRPC data structure.
 *
 * GstTensorSrcGRPC inherits GstPushSrcGRPC.
 */
struct _GstTensorSrcGRPC
{
  GstPushSrc element; /**< parent class object */

  /** Properties saved */
  gboolean silent;      /**< true to print minimized log */
  gboolean server;      /**< true to enable server mode */
  gint port;      /**< gRPC server port number */
  gchar *host;    /**< gRPC server host name */
  guint out;

  /** Working variables */
  GstDataQueue *queue; /**< data queue to hold input data */
  GstTensorsConfig config;

  /** private data */
  void *priv;
};

/**
 * @brief GstTensorSrcGRPCClass data structure.
 *
 * GstTensorSrcGRPC inherits GstPushSrc.
 */
struct _GstTensorSrcGRPCClass
{
  GstPushSrcClass parent_class; /**< inherits class object */
};

/**
 * @brief Function to get type of tensor_src_grpc.
 */
GType gst_tensor_src_grpc_get_type (void);

G_END_DECLS
#endif /** __GST_TENSOR_SRC_GRPC_H__ */
