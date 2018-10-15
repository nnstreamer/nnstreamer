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
 * @file	gsttensormux.h
 * @date	03 July 2018
 * @brief	GStreamer plugin to mux tensors (as a filter for other general neural network filters)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GST_TENSOR_MUX_H__
#define __GST_TENSOR_MUX_H__

#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <tensor_common.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_MUX (gst_tensor_mux_get_type ())
#define GST_TENSOR_MUX(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_TENSOR_MUX, GstTensorMux))
#define GST_TENSOR_MUX_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_TENSOR_MUX, GstTensorMuxClass))
#define GST_TENSOR_MUX_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_TENSOR_MUX, GstTensorMuxClass))
#define GST_IS_TENSOR_MUX(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_MUX))
#define GST_IS_TENSOR_MUX_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_MUX))
#define GST_TENSOR_MUX_CAST(obj)((GstTensorMux*)(obj))
typedef struct _GstTensorMux GstTensorMux;
typedef struct _GstTensorMuxClass GstTensorMuxClass;

typedef struct
{
  GstCollectData collect;
  GstClockTime pts_timestamp;
  GstClockTime dts_timestamp;
  GstPad *pad;
} GstTensorMuxPadData;

/**
 * @brief Tensor Muxer data structure
 */
struct _GstTensorMux
{
  GstElement element;

  gboolean silent;
  GstPad *srcpad;

  GstCollectPads *collect;
  gboolean negotiated;
  gboolean need_segment;
  gboolean need_stream_start;
  gboolean send_stream_start;

  GstTensorsConfig tensors_config; /**< output tensors info */
};

/**
 * @brief GstTensroMuxClass inherits GstElementClass
 */
struct _GstTensorMuxClass
{
  GstElementClass parent_class;
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_mux_get_type (void);

G_END_DECLS
#endif  /** __GST_TENSOR_MUX_H__ **/
