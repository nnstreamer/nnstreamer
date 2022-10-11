/* GStreamer
 *
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
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

/**
 * @file	gsttensor_trainsink.h
 * @date	11 October 2022
 * @brief	GStreamer plugin to train tensor data using NN Frameworks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	nnfw <nnfw@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_TRAINSINK_H__
#define __GST_TENSOR_TRAINSINK_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_TRAINSINK \
  (gst_tensor_trainsink_get_type())
#define GST_TENSOR_TRAINSINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_TRAINSINK,GstTensorTrainSink))
#define GST_TENSOR_TRAINSINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_TRAINSINK,GstTensorTrainSinkClass))
#define GST_IS_TENSOR_TRAINSINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_TRAINSINK))
#define GST_IS_TENSOR_TRAINSINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_TRAINSINK))
#define GST_TENSOR_TRAINSINK_CAST(obj) ((GstTensorTrainSink *)obj)
typedef struct _GstTensorTrainSink GstTensorTrainSink;
typedef struct _GstTensorTrainSinkClass GstTensorTrainSinkClass;

/**
 * @brief GstTensorTrainSink data structure
 *
 * The opaque #GstTensorTrainSink data structure.
 */
struct _GstTensorTrainSink
{
  GstBaseSink element;
  int dump;
};

/**
 * @brief GstTensorTrainSinkClass data structure.
 *
 * GstTensorTrainSinkClass inherits GstBaseSink.
 */
struct _GstTensorTrainSinkClass
{
  GstBaseSinkClass parent_class;
};

G_GNUC_INTERNAL GType gst_tensor_trainsink_get_type (void);

G_END_DECLS
#endif /* __GST_TENSOR_TRAINSINK_H__ */
