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
 * @file	tensor_reposink.h
 * @date	19 Nov 2018
 * @brief	GStreamer plugin to handle tensor repository
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_REPOSINK_H_
#define __GST_TENSOR_REPOSINK_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <tensor_repo.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_REPOSINK \
  (gst_tensor_reposink_get_type())
#define GST_TENSOR_REPOSINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_REPOSINK,GstTensorRepoSink))
#define GST_TENSOR_REPOSINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_REPOSINK,GstTensorRepoSinkClass))
#define GST_IS_TENSOR_REPOSINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_REPOSINK))
#define GST_IS_TENSOR_REPOSINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_REPOSINK))
typedef struct _GstTensorRepoSink GstTensorRepoSink;
typedef struct _GstTensorRepoSinkClass GstTensorRepoSinkClass;

/**
 * @brief GstTensorRepoSink data structure.
 *
 * GstTensorRepoSink inherits GstBaseSink.
 */
struct _GstTensorRepoSink
{
  GstBaseSink element;

  gboolean silent;
  gboolean emit_signal;
  guint signal_rate;
  GstClockTime last_render_time;
  GstCaps *in_caps;
  GstTensorData data;
  guint myid;

};

/**
 * @brief GstTensorRepoSinkClass data structure.
 *
 * GstTensorRepoSink inherits GstBaseSink.
 */
struct _GstTensorRepoSinkClass
{
  GstBaseSinkClass parent_class;

  void (*new_data) (GstElement * element, GstBuffer * buffer);
  void (*stream_start) (GstElement * element);
  void (*eos) (GstElement * element);
};

/**
 * @brief Function to get type of tensor_reposink.
 */
GType gst_tensor_reposink_get_type (void);

G_END_DECLS
#endif /** __GST_TENSOR_REPOSINK_H__ */
