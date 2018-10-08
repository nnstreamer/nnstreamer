/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_load.c
 * @date	24 Jul 2018
 * @brief	GStreamer plugin to convert other/tensorsave to other/tensor(s)
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GST_TENSOR_LOAD_H__
#define __GST_TENSOR_LOAD_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>

G_BEGIN_DECLS
/* #defines don't like whitespacey bits */
#define GST_TYPE_TENSOR_LOAD \
  (gst_tensor_load_get_type())
#define GST_TENSOR_LOAD(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_LOAD,GstTensor_Load))
#define GST_TENSOR_LOAD_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_LOAD,GstTensor_LoadClass))
#define GST_IS_TENSOR_LOAD(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_LOAD))
#define GST_IS_TENSOR_LOAD_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_LOAD))
#define GST_TENSOR_LOAD_CAST(obj)  ((GstTensor_Load *)(obj))
typedef struct _GstTensor_Load GstTensor_Load;

typedef struct _GstTensor_LoadClass GstTensor_LoadClass;

/**
 * @brief Internal data structure for tensor_load instances.
 */
struct _GstTensor_Load
{
  GstBaseTransform element;     /**< This is the parent object */

  /* For Tensor */
  gboolean silent;      /**< True if logging is minimized */

  guint num_tensors;    /**< Number of tensors in each frame */
  tensor_dim *dims;     /**< Array of tensor_dim, [num_tensors] */
  tensor_type *types;   /**< Array of tensor_type, [num_tensors] */
  gint *ranks;          /**< Array of rank, [num_tensors] */
  gint framerate_numerator;     /**< framerate is in fraction, which is numerator/denominator */
  gint framerate_denominator;   /**< framerate is in fraction, which is numerator/denominator */
  gsize frameSize;        /**< Size of a frame in # bytes */
};

/**
 * @brief GstTensor_LoadClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoLoad.
 * Note that GstTensor_LoadClass is a concrete class; thus we need to look at both.
 */
struct _GstTensor_LoadClass
{
  GstBaseTransformClass parent_class;   /**< Inherits GstBaseTransformClass */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_load_get_type (void);

G_END_DECLS
#endif /* __GST_TENSOR_LOAD_H__ */
