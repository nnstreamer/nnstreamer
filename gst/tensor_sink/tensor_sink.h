/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Samsung Electronics Co., Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 */

/**
 * @file	tensor_sink.h
 * @date	15 June 2018
 * @brief	GStreamer plugin to handle tensor stream
 * @see		http://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_SINK_H__
#define __GST_TENSOR_SINK_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_SINK \
  (gst_tensor_sink_get_type())
#define GST_TENSOR_SINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_SINK,GstTensorSink))
#define GST_TENSOR_SINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_SINK,GstTensorSinkClass))
#define GST_IS_TENSOR_SINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_SINK))
#define GST_IS_TENSOR_SINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_SINK))

typedef struct _GstTensorSink GstTensorSink;
typedef struct _GstTensorSinkClass GstTensorSinkClass;

/**
 * @brief GstTensorSink data structure.
 *
 * GstTensorSink inherits GstBaseSink.
 */
struct _GstTensorSink
{
  GstBaseSink element; /**< parent object */

  GMutex mutex; /**< mutex for processing */
  gboolean silent; /**< true to print minimized log */
  gboolean emit_signal; /**< true to emit signal for new data, eos */
  guint signal_rate; /**< new data signals per second */
  GstClockTime last_render_time; /**< buffer rendered time */
  GstCaps *in_caps; /**< received caps */
};

/**
 * @brief GstTensorSinkClass data structure.
 *
 * GstTensorSink inherits GstBaseSink.
 */
struct _GstTensorSinkClass
{
  GstBaseSinkClass parent_class; /**< parent class */

  /** signals */
  void (*new_data) (GstElement * element, GstBuffer * buffer); /**< signal when new data received */
  void (*stream_start) (GstElement * element); /**< signal when stream started */
  void (*eos) (GstElement * element); /**< signal when end of stream reached */
};

/**
 * @brief Function to get type of tensor_sink.
 */
GType gst_tensor_sink_get_type (void);

G_END_DECLS

#endif /** __GST_TENSOR_SINK_H__ */
