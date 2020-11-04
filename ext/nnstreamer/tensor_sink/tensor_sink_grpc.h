/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Sink_gRPC
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file	tensor_sink_grpc.h
 * @date	22 Oct 2020
 * @brief	GStreamer plugin to support gRPC tensor sink
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Dongju Chae <dongju.chae@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_SINK_GRPC_H__
#define __GST_TENSOR_SINK_GRPC_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

#include <nnstreamer_protobuf_grpc.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_SINK_GRPC \
  (gst_tensor_sink_grpc_get_type())
#define GST_TENSOR_SINK_GRPC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_SINK_GRPC,GstTensorSinkGRPC))
#define GST_TENSOR_SINK_GRPC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_SINK_GRPC,GstTensorSinkGRPCClass))
#define GST_IS_TENSOR_SINK_GRPC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_SINK_GRPC))
#define GST_IS_TENSOR_SINK_GRPC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_SINK_GRPC))

typedef struct _GstTensorSinkGRPC GstTensorSinkGRPC;
typedef struct _GstTensorSinkGRPCClass GstTensorSinkGRPCClass;

typedef enum {
  GST_TENSOR_SINK_GRPC_CONFIGURED = (GST_ELEMENT_FLAG_LAST << 0),
  GST_TENSOR_SINK_GRPC_STARTED = (GST_ELEMENT_FLAG_LAST << 1),
} GstTensorSinkGRPCFlags;

/**
 * @brief GstTensorSinkGRPC data structure.
 *
 * GstTensorSinkGRPC inherits GstPushSinkGRPC.
 */
struct _GstTensorSinkGRPC
{
  GstBaseSink element; /**< parent class object */

  /** Properties saved */
  gboolean silent;      /**< true to print minimized log */
  gboolean server;      /**< true to enable server mode */
  gint port;            /**< gRPC server port number */
  gchar *host;          /**< gRPC server host name */
  guint out;            /**< number of output messages */

  /** Working variables */
  GstTensorsConfig config;

  /** private data */
  void *priv;
};

/**
 * @brief GstTensorSinkGRPCClass data structure.
 *
 * GstTensorSinkGRPC inherits GstBaseSink.
 */
struct _GstTensorSinkGRPCClass
{
  GstBaseSinkClass parent_class; /**< inherits class object */
};

/**
 * @brief Function to get type of tensor_sink_grpc.
 */
GType gst_tensor_sink_grpc_get_type (void);

G_END_DECLS

#endif /** __GST_TENSOR_SINK_GRPC_H__ */
