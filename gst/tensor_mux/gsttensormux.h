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
 * @bug         No known bugs
 *
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GST_TENSOR_MUX_H__
#define __GST_TENSOR_MUX_H__

#include <gst/gst.h>
#include <tensor_common.h>
#include <tensor_meta.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_MUX (gst_tensor_mux_get_type ())
#define GST_TENSOR_MUX(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_TENSOR_MUX, GstTensorMux))
#define GST_TENSOR_MUX_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_TENSOR_MUX, GstTensorMuxClass))
#define GST_TENSOR_MUX_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_TENSOR_MUX, GstTensorMuxClass))
#define GST_IS_TENSOR_MUX(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_MUX))
#define GST_IS_TENSOR_MUX_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_MUX))
#define GST_TENSOR_MUX_CAST(obj)((GstTensorMux*)(obj))

typedef struct _GstTensorMux GstTensorMux;
typedef struct _GstTensorMuxClass GstTensorMuxClass;

/**
 * @brief Tensor Muxer pad private data structure
 */
typedef struct
{
  gboolean have_timestamp_offset;
  guint timestamp_offset;

  GstSegment segment;

  gboolean done;
  gboolean priority;
  gint nth;
} GstTensorMuxPadPrivate;


/**
 * @brief Tensor Muxer data structure
 */
struct _GstTensorMux
{
  GstElement element;

  guint64 byte_count;
  gboolean silent;
  GstPad *srcpad;
  GstPad *last_pad;
  GstBuffer *outbuffer;

  GString *dimensions;
  guint32 num_tensors;
  gint rank;
  GString *types;
  gint framerate_numerator;
  gint framerate_denominator;
  GstClockTime last_stop;
  gboolean send_stream_start;
};

/**
 * @brief GstTensroMuxClass inherits GstElementClass
 */
struct _GstTensorMuxClass {
  GstElementClass parent_class;
  /** gboolean (*src_event) (GstTensorMux *tensor_mux, GstEvent *event); */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_mux_get_type (void);

G_END_DECLS

#endif  /** __GST_TENSOR_MUX_H__ **/
