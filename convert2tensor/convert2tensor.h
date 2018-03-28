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
 * @file	convert2tensor.c
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert media types to tensors (as a filter for other general neural network filters)
 *
 *                Be careful: this filter assumes that the user has attached
 *               rawvideoparser as a preprocessor for this filter so that
 *               the incoming buffer is nicely aligned in the array of
 *               uint8[RGB][height][width].
 *
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 */

#ifndef __GST_CONVERT2TENSOR_H__
#define __GST_CONVERT2TENSOR_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video-info.h>
#include <gst/video/video-frame.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_CONVERT2TENSOR \
  (gst_convert2tensor_get_type())
#define GST_CONVERT2TENSOR(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CONVERT2TENSOR,GstConvert2Tensor))
#define GST_CONVERT2TENSOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CONVERT2TENSOR,GstConvert2TensorClass))
#define GST_IS_CONVERT2TENSOR(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CONVERT2TENSOR))
#define GST_IS_CONVERT2TENSOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CONVERT2TENSOR))
#define GST_CONVERT2TENSOR_CAST(obj)  ((GstConvert2Tensor *)(obj))

typedef struct _GstConvert2Tensor GstConvert2Tensor;

typedef struct _GstConvert2TensorClass GstConvert2TensorClass;

#define GST_CONVERT2TENSOR_TENSOR_RANK_LIMIT	(4)
typedef enum _tensor_type {
  _C2T_INT32 = 0,
  _C2T_UINT32,
  _C2T_INT16,
  _C2T_UINT16,
  _C2T_INT8,
  _C2T_UINT8,
  _C2T_FLOAT64,
  _C2T_FLOAT32,

  _C2T_END,
} tensor_type;
typedef enum _media_type {
  _C2T_VIDEO = 0,
  _C2T_AUDIO, /* Not Supported Yet */
  _C2T_STRING, /* Not Supported Yet */

  _C2T_MEDIA_END,
} media_type;
struct _GstConvert2Tensor
{
  GstBaseTransform element;	/**< This is the parent object */

  /* For transformer */
  gboolean negotiated; /* When this is %TRUE, tensor metadata must be set */
  media_type input_media_type;
  union {
    GstVideoInfo video;
    /* @TODO: Add other media types */
  } in_info;

  /* For Tensor */
  gboolean silent;	/**< True if logging is minimized */
  gboolean tensorConfigured;	/**< True if already successfully configured tensor metadata */
  gint rank;		/**< Tensor Rank (# dimensions) */
  gint dimension[GST_CONVERT2TENSOR_TENSOR_RANK_LIMIT];
      /**< Dimensions. We support up to 4th ranks.
       *  @caution The first dimension is always 4 x N.
       **/
  tensor_type type;		/**< Type of each element in the tensor. User must designate this. Otherwise, this is UINT8 for video/x-raw byte stream */
  gint framerate_numerator;	/**< framerate is in fraction, which is numerator/denominator */
  gint framerate_denominator;	/**< framerate is in fraction, which is numerator/denominator */
  gsize tensorFrameSize;
};
const unsigned int GstConvert2TensorDataSize[] = {
        [_C2T_INT32] = 4,
        [_C2T_UINT32] = 4,
        [_C2T_INT16] = 2,
        [_C2T_UINT16] = 2,
        [_C2T_INT8] = 1,
        [_C2T_UINT8] = 1,
        [_C2T_FLOAT64] = 8,
        [_C2T_FLOAT32] = 4,
};
const gchar* GstConvert2TensorDataTypeName[] = {
        [_C2T_INT32] = "int32",
        [_C2T_UINT32] = "uint32",
        [_C2T_INT16] = "int16",
        [_C2T_UINT16] = "uint16",
        [_C2T_INT8] = "int8",
        [_C2T_UINT8] = "uint8",
        [_C2T_FLOAT64] = "float64",
        [_C2T_FLOAT32] = "float32",
};

/*
 * GstConvert2TensorClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoConverter.
 * Note that GstConvert2TensorClass is a concrete class; thus we need to look at both.
 */
struct _GstConvert2TensorClass 
{
  GstBaseTransformClass parent_class;
};

GType gst_convert2tensor_get_type (void);

G_END_DECLS

#endif /* __GST_CONVERT2TENSOR_H__ */
