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
 * @file	gsttensorsplit.h
 * @date	27 Aug 2018
 * @brief	GStreamer plugin to split tensor (as a filter for other general neural network filters)
 * @bug         No known bugs
 *
 * @see		http://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GST_TENSOR_SPLIT_H__
#define __GST_TENSOR_SPLIT_H__

#include <gst/gst.h>
#include <tensor_common.h>
#include <tensor_meta.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_SPLIT (gst_tensor_split_get_type ())
#define GST_TENSOR_SPLIT(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_TENSOR_SPLIT, GstTensorSplit))
#define GST_TENSOR_SPLIT_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_TENSOR_SPLIT, GstTensorSplitClass))
#define GST_TENSOR_SPLIT_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_TENSOR_SPLIT, GstTensorSplitClass))
#define GST_IS_TENSOR_SPLIT(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_SPLIT))
#define GST_IS_TENSOR_SPLIT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_SPLIT))
#define GST_TENSOR_SPLIT_CAST(obj)((GstTensorSplit*)(obj))
typedef struct _GstTensorSplit GstTensorSplit;
typedef struct _GstTensorSplitClass GstTensorSplitClass;

typedef struct
{
  GstPad *pad;
  GstClockTime last_ts;
  GstFlowReturn last_ret;
  gboolean discont;
  gint nth;
} GstTensorPad;

/**
 * @brief Tensor Spliter data structure
 */
struct _GstTensorSplit
{
  GstElement element;

  gboolean silent;
  GstPad *sinkpad;
  GSList *srcpads;
  guint32 num_tensors;
  guint32 num_srcpads;
  GList *tensorpick;
  GArray *tensorseg;
  gboolean have_group_id;
  guint group_id;
  GstTensorConfig sink_tensor_conf;
};

/**
 * @brief GstTensorSplitClass inherits GstElementClass
 */
struct _GstTensorSplitClass
{
  GstElementClass parent_class;
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_split_get_type (void);

G_END_DECLS
#endif  /** __GST_TENSOR_SPLIT_H__ **/
