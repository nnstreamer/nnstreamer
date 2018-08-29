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
 * @file	tensor_aggregator.h
 * @date	29 August 2018
 * @brief	GStreamer plugin to aggregate tensor stream
 * @see		https://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_AGGREGATOR_H__
#define __GST_TENSOR_AGGREGATOR_H__

#include <gst/gst.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_AGGREGATOR \
  (gst_tensor_aggregator_get_type())
#define GST_TENSOR_AGGREGATOR(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_AGGREGATOR,GstTensorAggregator))
#define GST_TENSOR_AGGREGATOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_AGGREGATOR,GstTensorAggregatorClass))
#define GST_IS_TENSOR_AGGREGATOR(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_AGGREGATOR))
#define GST_IS_TENSOR_AGGREGATOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_AGGREGATOR))

typedef struct _GstTensorAggregator GstTensorAggregator;
typedef struct _GstTensorAggregatorClass GstTensorAggregatorClass;

/**
 * @brief GstTensorAggregator data structure.
 */
struct _GstTensorAggregator
{
  GstElement element; /**< parent object */

  GstPad *sinkpad; /**< sink pad */
  GstPad *srcpad; /**< src pad */

  gboolean silent; /**< true to print minimized log */
  guint frames_in; /**< number of frames in input buffer */
  guint frames_out; /**< number of frames in output buffer */
  guint frames_flush; /**< number of frames to flush */

  GstAdapter *adapter; /**< adapt incoming tensor */

  gboolean tensor_configured; /**< True if already successfully configured tensor metadata */
  GstTensorConfig in_config; /**< input tensor info */
  GstTensorConfig out_config; /**< output tensor info */
};

/**
 * @brief GstTensorAggregatorClass data structure.
 */
struct _GstTensorAggregatorClass
{
  GstElementClass parent_class; /**< parent class */
};

/**
 * @brief Function to get type of tensor_aggregator.
 */
GType gst_tensor_aggregator_get_type (void);

G_END_DECLS

#endif /** __GST_TENSOR_AGGREGATOR_H__ */
