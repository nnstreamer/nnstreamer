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
 * @see		https://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GST_TENSOR_CONVERTER_H__
#define __GST_TENSOR_CONVERTER_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video-info.h>
#include <gst/audio/audio-info.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_CONVERTER \
  (gst_tensor_converter_get_type())
#define GST_TENSOR_CONVERTER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_CONVERTER,GstTensorConverter))
#define GST_TENSOR_CONVERTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_CONVERTER,GstTensorConverterClass))
#define GST_IS_TENSOR_CONVERTER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_CONVERTER))
#define GST_IS_TENSOR_CONVERTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_CONVERTER))
#define GST_TENSOR_CONVERTER_CAST(obj)  ((GstTensorConverter *)(obj))

typedef struct _GstTensorConverter GstTensorConverter;
typedef struct _GstTensorConverterClass GstTensorConverterClass;

/**
 * @brief Internal data structure for tensor_converter instances.
 */
struct _GstTensorConverter
{
  GstBaseTransform element; /**< This is the parent object */

  /** For transformer */
  gboolean negotiated; /**< %TRUE if tensor metadata is set */
  union
  {
    GstVideoInfo video; /**< video-info of the input media stream */
    GstAudioInfo audio; /**< audio-info of the input media stream */
    /** @todo: Add other media types */
  } in_info; /**< media input stream info union. will support audio/text later */
  gboolean removePadding; /**< If TRUE, zero-padding must be removed during transform */
  gboolean disableInPlace; /**< If TRUE, In place mode is disabled */
  gboolean silent; /**< True if logging is minimized */
  guint frames_per_buffer; /**< frame count per buffer */

  /** For Tensor */
  gboolean tensor_configured; /**< True if already successfully configured tensor metadata */
  GstTensorConfig tensor_config; /**< configured tensor info */
};

/**
 * @brief GstTensorConverterClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoConverter.
 * Note that GstTensorConverterClass is a concrete class; thus we need to look at both.
 */
struct _GstTensorConverterClass
{
  GstBaseTransformClass parent_class; /**< Inherits GstBaseTransformClass */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_converter_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_CONVERTER_H__ */
