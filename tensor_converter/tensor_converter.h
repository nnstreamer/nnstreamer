/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
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
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * @file	tensor_converter.c
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert media types to tensors (as a filter for other general neural network filters)
 *
 *                Be careful: this filter assumes that the user has attached
 *               other GST converters as a preprocessor for this filter so that
 *               the incoming buffer is nicely aligned in the array of
 *               uint8[height][width][RGB]. Note that if rstride=RU4, you need
 *               to add the case in "remove_stride_padding_per_row".
 *
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 */

#ifndef __GST_TENSOR_CONVERTER_H__
#define __GST_TENSOR_CONVERTER_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video-info.h>
#include <gst/video/video-frame.h>
#include <tensor_common.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_TENSOR_CONVERTER \
  (gst_tensor_converter_get_type())
#define GST_TENSOR_CONVERTER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_CONVERTER,GstTensor_Converter))
#define GST_TENSOR_CONVERTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_CONVERTER,GstTensor_ConverterClass))
#define GST_IS_TENSOR_CONVERTER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_CONVERTER))
#define GST_IS_TENSOR_CONVERTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_CONVERTER))
#define GST_TENSOR_CONVERTER_CAST(obj)  ((GstTensor_Converter *)(obj))

typedef struct _GstTensor_Converter GstTensor_Converter;

typedef struct _GstTensor_ConverterClass GstTensor_ConverterClass;

/**
 * @brief Internal data structure for tensor_converter instances.
 */
struct _GstTensor_Converter
{
  GstBaseTransform element;	/**< This is the parent object */

  /* For transformer */
  gboolean negotiated; /**< %TRUE if tensor metadata is set */
  media_type input_media_type; /**< Denotes the input media stream type */
  union {
    GstVideoInfo video; /**< video-info of the input media stream */
    /* @TODO: Add other media types */
  } in_info; /**< media input stream info union. will support audio/text later */

  /* For Tensor */
  gboolean silent;	/**< True if logging is minimized */
  gboolean tensorConfigured;	/**< True if already successfully configured tensor metadata */
  gint rank;		/**< Tensor Rank (# dimensions) */
  gint dimension[NNS_TENSOR_RANK_LIMIT]; /**< Dimensions. We support up to 4th ranks.  **/
  tensor_type type;		/**< Type of each element in the tensor. User must designate this. Otherwise, this is UINT8 for video/x-raw byte stream */
  gint framerate_numerator;	/**< framerate is in fraction, which is numerator/denominator */
  gint framerate_denominator;	/**< framerate is in fraction, which is numerator/denominator */
  gsize tensorFrameSize;
};

/*
 * @brief GstTensor_ConverterClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoConverter.
 * Note that GstTensor_ConverterClass is a concrete class; thus we need to look at both.
 */
struct _GstTensor_ConverterClass 
{
  GstBaseTransformClass parent_class;	/**< Inherits GstBaseTransformClass */
};

/*
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_converter_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_CONVERTER_H__ */
