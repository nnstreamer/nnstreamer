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
 * @file	tensor_repopush.h
 * @date	19 Nov 2018
 * @brief	GStreamer plugin to handle tensor repository
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_REPOPUSH_H_
#define __GST_TENSOR_REPOPUSH_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <tensor_repo.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_REPOPUSH \
  (gst_tensor_repopush_get_type())
#define GST_TENSOR_REPOPUSH(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_REPOPUSH,GstTensorRepoPush))
#define GST_TENSOR_REPOPUSH_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_REPOPUSH,GstTensorRepoPushClass))
#define GST_IS_TENSOR_REPOPUSH(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_REPOPUSH))
#define GST_IS_TENSOR_REPOPUSH_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_REPOPUSH))
typedef struct _GstTensorRepoPush GstTensorRepoPush;
typedef struct _GstTensorRepoPushClass GstTensorRepoPushClass;

/**
 * @brief GstTensorRepoPush data structure.
 *
 * GstTensorRepoPush inherits GstBaseSink.
 */
struct _GstTensorRepoPush
{
  GstBaseSink element;

  gboolean silent;
  guint signal_rate;
  GstClockTime last_render_time;
  GstCaps *in_caps;
  GstTensorData data;
  guint myid;

};

/**
 * @brief GstTensorRepoPushClass data structure.
 *
 * GstTensorRepoPush inherits GstBaseSink.
 */
struct _GstTensorRepoPushClass
{
  GstBaseSinkClass parent_class;
};

/**
 * @brief Function to get type of tensor_repopush.
 */
GType gst_tensor_repopush_get_type (void);

G_END_DECLS
#endif /** __GST_TENSOR_REPOPUSH_H__ */
