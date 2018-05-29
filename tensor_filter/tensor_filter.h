/*
 * GStreamer Tensor_Filter
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
 * @file	tensor_filter.h
 * @date	24 May 2018
 * @brief	GStreamer plugin to use general neural network frameworks as filters
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * @TODO TBD: Should we disable "in-place" mode? (what if output size > input size?)
 */

#ifndef __GST_TENSOR_FILTER_H__
#define __GST_TENSOR_FILTER_H__

#include <stdint.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_TENSOR_FILTER \
  (gst_tensor_filter_get_type())
#define GST_TENSOR_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_FILTER,GstTensor_Filter))
#define GST_TENSOR_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_FILTER,GstTensor_FilterClass))
#define GST_IS_TENSOR_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_FILTER))
#define GST_IS_TENSOR_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_FILTER))
#define GST_TENSOR_FILTER_CAST(obj)  ((GstTensor_Filter *)(obj))

typedef struct _GstTensor_Filter GstTensor_Filter;

typedef struct _GstTensor_FilterClass GstTensor_FilterClass;

/**
 * @brief NN Frameworks
 */
typedef enum _nnfw_type {
  _T_F_UNDEFINED = 0, /* Not defined or supported. Cannot proceed in this status */

  _T_F_CUSTOM, /* NYI. Custom other/tensor -> other/tensor shared object (dysym) */
  _T_F_TENSORFLOW_LITE, /* NYI */
  _T_F_TENSORFLOW, /* NYI */
  _T_F_CAFFE2, /* NYI */

  _T_F_NNFW_END,
} nnfw_type;

extern const char* nnfw_names[];

/**
 * @brief NN Framework Support Status
 *
 * TRUE: Supported
 * FALSE: Scaffoldings are there, but not supported, yet. (NYI)
 */
static const gboolean nnfw_support_status[] = {
  FALSE,

  FALSE,
  FALSE,
  FALSE,
  FALSE,

  FALSE,
};

/**
 * @brief Internal data structure for tensor_filter instances.
 */
struct _GstTensor_Filter
{
  GstBaseTransform element;	/**< This is the parent object */

  gboolean silent;
  gboolean inputConfigured;
  gboolean outputConfigured;
  nnfw_type nnfw;
  gchar *modelFilename;

  uint32_t inputDimension[NNS_TENSOR_RANK_LIMIT];
  int inputType;
  uint32_t outputDimension[NNS_TENSOR_RANK_LIMIT];
  int outputType;

  void *privateData; /**< NNFW plugin's private data is stored here */
};

/*
 * @brief GstTensor_FilterClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoConverter.
 * Note that GstTensor_FilterClass is a concrete class; thus we need to look at both.
 */
struct _GstTensor_FilterClass
{
  GstBaseTransformClass parent_class;	/**< Inherits GstBaseTransformClass */
};

/*
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_filter_get_type (void);

struct _GstTensor_Filter_Framework
{
  gchar *name; /**< Name of the neural network framework, searchable by FRAMEWORK property */
  gboolean allow_in_place; /**< TRUE if InPlace transfer of input-to-output is allowed. Not supported in main, yet */
  int (*invoke_NN)(GstTensor_Filter *filter, void *inputptr, void *outputptr); /**< Mandatory callback. Invoke the given network model. */
  int (*getInputDimension)(GstTensor_Filter *filter, uint32_t *inputDimension); /**< Optional. Set NULL if not supported. Get dimension of input tensor */
  int (*getOutputDimension)(GstTensor_Filter *filter, uint32_t *outputDimension); /**< Optional. Set NULL if not supported. Get dimension of output tensor */
};
typedef struct _GstTensor_Filter_Framework GstTensor_Filter_Framework;

extern GstTensor_Filter_Framework NNS_support_tensorflow_lite;

extern GstTensor_Filter_Framework *tensor_filter_supported[];

G_END_DECLS

#endif /* __GST_TENSOR_FILTER_H__ */
