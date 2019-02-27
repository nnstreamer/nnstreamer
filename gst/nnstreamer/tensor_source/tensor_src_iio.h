/**
 * GStreamer Tensor_Src_IIO
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
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
 */

/**
 * @file	tensor_src_iio.h
 * @date	26 Feb 2019
 * @brief	GStreamer plugin to support linux IIO as tensor(s)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_SRC_IIO_H__
#define __GST_TENSOR_SRC_IIO_H__

#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>
#include <tensor_common.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_SRC_IIO \
  (gst_tensor_src_iio_get_type())
#define GST_TENSOR_SRC_IIO(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_SRC_IIO,GstTensorSrcIIO))
#define GST_TENSOR_SRC_IIO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_SRC_IIO,GstTensorSrcIIOClass))
#define GST_IS_TENSOR_SRC_IIO(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_SRC_IIO))
#define GST_IS_TENSOR_SRC_IIO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_SRC_IIO))
typedef struct _GstTensorSrcIIO GstTensorSrcIIO;
typedef struct _GstTensorSrcIIOClass GstTensorSrcIIOClass;

/**
 * @brief GstTensorSrcIIO data structure.
 *
 * GstTensorSrcIIO inherits GstBaseSrcIIO.
 */
struct _GstTensorSrcIIO
{
  GstBaseSrc element; /**< parent class object */

  /* gstreamer related properties */
  GMutex mutex; /**< mutex for processing */

  /* linux IIO related properties */
  gchar *device_name; /**< IIO device name */
  GList *triggers; /**< triggers for the IIO device */
  GList *channels_enabled; /**< channels to be enabled */
  guint buffer_capacity; /**< size of the buffer */
  guint sampling_frequency; /**< sampling frequncy for the device */
};

/**
 * @brief GstTensorSrcIIOClass data structure.
 *
 * GstTensorSrcIIO inherits GstBaseSrc.
 */
struct _GstTensorSrcIIOClass
{
  GstBaseSrcClass parent_class; /**< inherits class object */
};

/**
 * @brief Function to get type of tensor_src_iio.
 */
GType gst_tensor_src_iio_get_type (void);

G_END_DECLS
#endif /** __GST_TENSOR_SRC_IIO_H__ */
