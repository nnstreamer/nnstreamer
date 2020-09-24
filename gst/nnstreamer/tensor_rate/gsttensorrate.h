/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-Rate
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    gsttensorrate.h
 * @date    24 Sep 2020
 * @brief   GStreamer plugin to adjust tensor rate
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_RATE_H__
#define __GST_TENSOR_RATE_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include <tensor_common.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_RATE (gst_tensor_rate_get_type ())
#define GST_TENSOR_RATE(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_TENSOR_RATE, GstTensorRate))
#define GST_TENSOR_RATE_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_TENSOR_RATE, GstTensorRateClass))
#define GST_TENSOR_RATE_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_TENSOR_RATE, GstTensorRateClass))
#define GST_IS_TENSOR_RATE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_RATE))
#define GST_IS_TENSOR_RATE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_RATE))
#define GST_TENSOR_RATE_CAST(obj)((GstTensorRate*)(obj))
typedef struct _GstTensorRate GstTensorRate;
typedef struct _GstTensorRateClass GstTensorRateClass;

/**
 * @brief Tensor Rate data structure
 */
struct _GstTensorRate
{
  GstBaseTransform element;     /**< This is the parent object */

  GstBuffer *prevbuf;           /**< previous buffer */
  GstSegment segment;           /**< current segment */
  guint64 out_frame_count;      /**< number of frames output */

  /** Caps negotiation */
  gint from_rate_numerator;     /**< framerate numerator (From) */
  gint from_rate_denominator;   /**< framerate denominator (From) */

  gint to_rate_numerator;       /**< framerate numerator (To) */
  gint to_rate_denominator;     /**< framerate denominator (To) */

  /** Timestamp */
  guint64 base_ts;              /**< used in next_ts calculation */
  guint64 prev_ts;              /**< Previous buffer timestamp */
  guint64 next_ts;              /**< Timestamp of next buffer to output */
  guint64 last_ts;              /**< Timestamp of last input buffer */

  /** Properties */
  guint64 in, out, dup, drop;   /**< stat property */
  gint rate_n, rate_d;          /**< framerate property */
  gboolean silent;              /**< debug property */
  gboolean throttle;            /**< throttle property */
};

/**
 * @brief GstTensorRateClass inherits GstElementClass
 */
struct _GstTensorRateClass
{
  GstBaseTransformClass parent_class;   /**< Inherits GstBaseTransformClass */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_rate_get_type (void);

G_END_DECLS
#endif /* __GST_TENSOR_RATE_H__ */
