/**
 * GStreamer / NNStreamer tensor_decoder header
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
 * @file	tensordec.h
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert tensors to media types
 *
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GST_TENSORDEC_H__
#define __GST_TENSORDEC_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>
#include <nnstreamer_subplugin.h>
#include <nnstreamer_plugin_api_decoder.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_DECODER \
  (gst_tensordec_get_type())
#define GST_TENSOR_DECODER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_DECODER,GstTensorDec))
#define GST_TENSOR_DECODER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_DECODER,GstTensorDecClass))
#define GST_IS_TENSOR_DECODER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_DECODER))
#define GST_IS_TENSOR_DECODER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_DECODER))
#define GST_TENSOR_DECODER_CAST(obj)  ((GstTensorDec *)(obj))

typedef struct _GstTensorDec GstTensorDec;
typedef struct _GstTensorDecClass GstTensorDecClass;

#define TensorDecMaxOpNum (9)

/**
 * @brief Internal data structure for tensordec instances.
 */
struct _GstTensorDec
{
  GstBaseTransform element; /**< This is the parent object */

  /** For transformer */
  gboolean negotiated; /**< TRUE if tensor metadata is set */
  gboolean silent; /**< True if logging is minimized */
  guint output_type; /**< Denotes the output type */
  guint mode; /** Mode for tensor decoder "direct_video" or "image_labeling" or "bounding_boxes */
  gchar *option[TensorDecMaxOpNum]; /**< Assume we have two options */

  /** For Tensor */
  gboolean configured; /**< TRUE if already successfully configured tensor metadata */
  void *plugin_data;
  void (*cleanup_plugin_data)(void **pdata); /**< exit() of subplugin is registered here. If it's null, gfree(plugin_data) is used. */
  GstTensorsConfig tensor_config; /**< configured tensor info @todo support tensors in the future */

  const TensorDecDef *decoder; /**< Plugin object */
};

/**
 * @brief GstTensorDecClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoConverter.
 * Note that GstTensorDecClass is a concrete class; thus we need to look at both.
 */
struct _GstTensorDecClass
{
  GstBaseTransformClass parent_class; /**< Inherits GstBaseTransformClass */
};

/**
 * @brief Decoder Mode.
 */
typedef enum
{
  DECODE_MODE_PLUGIN,
  DECODE_MODE_UNKNOWN
} GstDecMode;

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensordec_get_type (void);

G_END_DECLS

#endif /* __GST_TENSORDEC_H__ */
