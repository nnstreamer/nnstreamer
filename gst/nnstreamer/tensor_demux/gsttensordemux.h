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
 * @file	gsttensordemux.h
 * @date	03 July 2018
 * @brief	GStreamer plugin to demux tensors (as a filter for other general neural network filters)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GST_TENSOR_DEMUX_H__
#define __GST_TENSOR_DEMUX_H__

#include <gst/gst.h>
#include <tensor_common.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_DEMUX (gst_tensor_demux_get_type ())
#define GST_TENSOR_DEMUX(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_TENSOR_DEMUX, GstTensorDemux))
#define GST_TENSOR_DEMUX_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_TENSOR_DEMUX, GstTensorDemuxClass))
#define GST_TENSOR_DEMUX_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_TENSOR_DEMUX, GstTensorDemuxClass))
#define GST_IS_TENSOR_DEMUX(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_DEMUX))
#define GST_IS_TENSOR_DEMUX_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_DEMUX))
#define GST_TENSOR_DEMUX_CAST(obj)((GstTensorDemux*)(obj))
typedef struct _GstTensorDemux GstTensorDemux;
typedef struct _GstTensorDemuxClass GstTensorDemuxClass;

typedef struct
{
  GstPad *pad;
  GstClockTime last_ts;
  GstFlowReturn last_ret;
  gint nth;
} GstTensorPad;

/**
 * @brief Tensor Muxer data structure
 */
struct _GstTensorDemux
{
  GstElement element;

  gboolean silent;
  GstPad *sinkpad;
  GSList *srcpads;
  guint32 num_srcpads;
  GList *tensorpick;
  gboolean have_group_id;
  guint group_id;

  GstTensorsConfig tensors_config; /**< input tensors info */
};

/**
 * @brief GstTensorDeMuxClass inherits GstElementClass
 */
struct _GstTensorDemuxClass
{
  GstElementClass parent_class;
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_demux_get_type (void);

G_END_DECLS
#endif  /** __GST_TENSOR_DEMUX_H__ **/
