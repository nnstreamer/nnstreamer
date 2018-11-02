/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 */
/**
 * @file	gsttensormerge.h
 * @date	03 July 2018
 * @brief	GStreamer plugin to merge tensors (as a filter for other general neural network filters)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GST_TENSOR_MERGE_H__
#define __GST_TENSOR_MERGE_H__

#include <gst/gst.h>
#include <tensor_common.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_MERGE (gst_tensor_merge_get_type ())
#define GST_TENSOR_MERGE(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_TENSOR_MERGE, GstTensorMerge))
#define GST_TENSOR_MERGE_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_TENSOR_MERGE, GstTensorMergeClass))
#define GST_TENSOR_MERGE_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_TENSOR_MERGE, GstTensorMergeClass))
#define GST_IS_TENSOR_MERGE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_MERGE))
#define GST_IS_TENSOR_MERGE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_MERGE))
#define GST_TENSOR_MERGE_CAST(obj)((GstTensorMerge*)(obj))
typedef struct _GstTensorMerge GstTensorMerge;
typedef struct _GstTensorMergeClass GstTensorMergeClass;

/**
 * @brief Tensor Merge time sync data for baspad mode
 */
typedef struct _tensor_time_sync_basepad {
  guint sink_id;
  guint duration;
} tensor_time_sync_basepad;

typedef enum
{
  GTT_LINEAR = 0,               /* Dimension Change. "dimchg" */
  GTT_END,
} tensor_merge_mode;

typedef enum
{
  LINEAR_FIRST = 0, 		/* CHANNEL */
  LINEAR_SECOND = 1,		/* WIDTH */
  LINEAR_THIRD = 2,		/* HEIGHT */
  LINEAR_FOURTH = 3,  		/* BATCH */
  LINEAR_END,
} tensor_merge_linear_mode;


/**
 * @brief Internal data structure for linear mode.
 */
typedef struct _tensor_merge_linear {
  tensor_merge_linear_mode direction;
} tensor_merge_linear;

/**
 * @brief Tensor Merge data structure
 */
struct _GstTensorMerge
{
  GstElement element;

  gboolean silent;
  tensor_time_sync_mode sync_mode;
  gchar *sync_option;
  union{
    tensor_time_sync_basepad data_basepad;
  };
  GstPad *srcpad;
  gchar *option;
  tensor_merge_mode mode;
  union{
    tensor_merge_linear data_linear;
  };

  gboolean loaded;
  GstCollectPads *collect;
  gboolean negotiated;
  gboolean need_segment;
  gboolean need_stream_start;
  gboolean send_stream_start;

  gboolean need_buffer;
  GstClockTime current_time;
  gboolean need_set_time;
  GstTensorsConfig tensors_config; /**< output tensors info */
};

/**
 * @brief GstTensorMergeClass inherits GstElementClass
 */
struct _GstTensorMergeClass
{
  GstElementClass parent_class;
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_merge_get_type (void);

G_END_DECLS
#endif  /** __GST_TENSOR_MERGE_H__ **/
