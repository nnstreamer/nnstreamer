/**
 * GStreamer Tensor_Filter
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	tensor_filter.h
 * @date	24 May 2018
 * @brief	GStreamer plugin to use general neural network frameworks as filters
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @todo TBD: Should we disable "in-place" mode? (what if output size > input size?)
 */

#ifndef __GST_TENSOR_FILTER_H__
#define __GST_TENSOR_FILTER_H__

#include <gst/gst.h>
#include <gst/gstinfo.h>
#include <gst/base/gstbasetransform.h>

#include "tensor_common.h"
#include "nnstreamer_subplugin.h"
#include "nnstreamer_plugin_api_filter.h"
#include "tensor_filter_common.h"

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_FILTER \
  (gst_tensor_filter_get_type())
#define GST_TENSOR_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_FILTER,GstTensorFilter))
#define GST_TENSOR_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_FILTER,GstTensorFilterClass))
#define GST_IS_TENSOR_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_FILTER))
#define GST_IS_TENSOR_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_FILTER))
#define GST_TENSOR_FILTER_CAST(obj)  ((GstTensorFilter *)(obj))

typedef struct _GstTensorFilter GstTensorFilter;
typedef struct _GstTensorFilterClass GstTensorFilterClass;

/**
 * @brief Internal data structure for tensor_filter instances.
 */
struct _GstTensorFilter
{
  GstBaseTransform element;     /**< This is the parent object */

  GstTensorFilterPrivate priv; /**< Internal properties for tensor-filter */

  GstClockTime prev_ts;  /**< previous timestamp */
  GstClockTimeDiff throttling_delay;  /**< throttling delay from tensor rate */
  GstClockTimeDiff throttling_accum;  /**< accumulated frame durations for throttling */
};

/**
 * @brief GstTensorFilterClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoConverter.
 * Note that GstTensorFilterClass is a concrete class; thus we need to look at both.
 */
struct _GstTensorFilterClass
{
  GstBaseTransformClass parent_class;   /**< Inherits GstBaseTransformClass */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_filter_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_FILTER_H__ */
