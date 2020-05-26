/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 */

/**
 * @file	tensor_converter.h
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert media types to tensors (as a filter for other general neural network filters)
 *
 *                Be careful: this filter assumes that the user has attached
 *               other GST converters as a preprocessor for this filter so that
 *               the incoming buffer is nicely aligned in the array of
 *               uint8[height][width][RGB]. Note that if rstride=RU4, you need
 *               to add the case in "remove_stride_padding_per_row".
 *
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_CONVERTER_H__
#define __GST_TENSOR_CONVERTER_H__

#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <tensor_common.h>
#include "nnstreamer_plugin_api_converter.h"

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

typedef struct _GstTensorConverter GstTensorConverter;
typedef struct _GstTensorConverterClass GstTensorConverterClass;

/**
 * @brief Internal data structure for tensor_converter instances.
 */
struct _GstTensorConverter
{
  GstElement element; /**< parent object */

  GstPad *sinkpad; /**< sink pad */
  GstPad *srcpad; /**< src pad */

  gboolean silent; /**< true to print minimized log */
  gboolean set_timestamp; /**< true to set timestamp when received a buffer with invalid timestamp */
  guint frames_per_tensor; /**< number of frames in output tensor */
  GstTensorInfo tensor_info; /**< data structure to get/set tensor info */

  GstAdapter *adapter; /**< adapt incoming media stream */

  media_type in_media_type; /**< incoming media type */
  /** ExternalConverter is used if in_media_type == _NNS_MEDIA_PLUGINS */
  const NNStreamerExternalConverter *externalConverter;

  gsize frame_size; /**< size of one frame */
  gboolean remove_padding; /**< If true, zero-padding must be removed */
  gboolean tensors_configured; /**< True if already successfully configured tensors metadata */
  GstTensorsConfig tensors_config; /**< output tensors info */

  gboolean have_segment; /**< True if received segment */
  gboolean need_segment; /**< True to handle seg event */
  GstSegment segment; /**< Segment, supposed time format */
  GstClockTime old_timestamp; /**< timestamp at prev buffer */
};

/**
 * @brief GstTensorConverterClass data structure.
 */
struct _GstTensorConverterClass
{
  GstElementClass parent_class; /**< parent class */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_converter_get_type (void);


G_END_DECLS

#endif /** __GST_TENSOR_CONVERTER_H__ */
