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

G_BEGIN_DECLS
#define GST_TYPE_TENSORDEC \
  (gst_tensordec_get_type())
#define GST_TENSORDEC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSORDEC,GstTensorDec))
#define GST_TENSORDEC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSORDEC,GstTensorDecClass))
#define GST_IS_TENSORDEC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSORDEC))
#define GST_IS_TENSORDEC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSORDEC))
#define GST_TENSORDEC_CAST(obj)  ((GstTensorDec *)(obj))
#define BOX_SIZE        4
#define DETECTION_MAX   1917
typedef struct _GstTensorDec GstTensorDec;
typedef struct _GstTensorDecClass GstTensorDecClass;
typedef struct _TensorDecDef TensorDecDef;

/**
 * @brief Data structure for image labeling info.
 */
typedef struct
{
  gchar *label_path; /**< label file path */
  GList *labels; /**< list of loaded labels */
  guint total_labels; /**< count of labels */
} Mode_image_labeling;

/**
 * @brief Data structure for boundig box info.
 */
typedef struct
{
  gchar *label_path; /**< label file path */
  GList *labels; /**< list of loaded labels */
  gchar *box_prior_path; /**< label file path */
  gfloat box_priors[BOX_SIZE][DETECTION_MAX];
  guint total_labels; /**< count of labels */
} Mode_boundig_boxes;

#define TensorDecMaxOpNum (3)
/**
 * @brief Internal data structure for tensordec instances.
 */
struct _GstTensorDec
{
  GstBaseTransform element; /**< This is the parent object */

  /** For transformer */
  gboolean negotiated; /**< TRUE if tensor metadata is set */
  gboolean add_padding; /**< If TRUE, zero-padding must be added during transform */
  gboolean silent; /**< True if logging is minimized */
  guint output_type; /**< Denotes the output type */
  guint mode; /** Mode for tensor decoder "direct_video" or "image_labeling" or "bounding_boxes */
  gchar *option[TensorDecMaxOpNum]; /**< Assume we have two options */

  /** For Tensor */
  gboolean configured; /**< TRUE if already successfully configured tensor metadata */
  void *plugin_data;
  void (*cleanup_plugin_data)(GstTensorDec *self); /**< exit() of subplugin is registered here. If it's null, gfree(plugin_data) is used. */
  GstTensorConfig tensor_config; /**< configured tensor info @todo support tensors in the future */
  Mode_image_labeling image_labeling;/** tensor decoder image labeling mode info */
  Mode_boundig_boxes bounding_boxes;/** tensor decoder image labeling mode info */

  TensorDecDef *decoder; /**< Plugin object */
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
 * @brief Output type.
 */
typedef enum
{
  OUTPUT_VIDEO,
  OUTPUT_AUDIO,
  OUTPUT_TEXT,
  OUTPUT_UNKNOWN
} GstDecMediaType;

/**
 * @brief Decoder Mode.
 */
typedef enum
{
  IMAGE_LABELING,
  BOUNDING_BOXES,
  DECODE_MODE_PLUGIN,
  DECODE_MODE_UNKNOWN
} GstDecMode;

/**
 * @brief Output type for each mode
 */
static const GstDecMediaType dec_output_type[] = {
  OUTPUT_VIDEO,
  OUTPUT_TEXT,
  OUTPUT_VIDEO,
  OUTPUT_UNKNOWN,
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensordec_get_type (void);


/*********************************************************
 * The followings are decoder-subplugin APIs             *
 *********************************************************/
/**
 * @brief Decoder definitions for different semantics of tensors
 *        This allows developers to create their own decoders.
 */
struct _TensorDecDef
{
  gchar *modename;
      /**< Unique decoder name. GST users choose decoders with mode="modename". */
  GstDecMediaType type;
      /**< Output media type. VIDEO/AUDIO/TEXT are supported */
  gboolean (*init) (GstTensorDec *self);
      /**< Object initialization for the decoder */
  void (*exit) (GstTensorDec *self);
      /**< Object destruction for the decoder */
  gboolean (*setOption) (GstTensorDec *self, int opNum, const gchar *param);
      /**< Process with the given options. It can be called repeatedly */
  GstCaps *(*getOutputDim) (GstTensorDec *self, const GstTensorConfig *config);
      /**< The caller should unref the returned GstCaps
        * Current implementation supports single-tensor only.
        * @todo WIP: support multi-tensor for input!!!
        */
  GstFlowReturn (*decode) (GstTensorDec *self, const GstTensorMemory *input,
      GstBuffer *outbuf);
      /**< outbuf must be allocated but empty (gst_buffer_get_size (outbuf) == 0).
        * Note that we support single-tensor (other/tensor) only!
        * @todo WIP: support multi-tensor for input!!!
        */
  gsize (*getTransformSize) (GstTensorDec *self, GstCaps *caps, gsize size, GstCaps *othercaps, GstPadDirection direction);
      /**< EXPERIMENTAL! @todo We are not ready to use this. This should be NULL or return 0 */
};

extern gboolean tensordec_probe (TensorDecDef *decoder);
extern void tensordec_exit (const gchar *name);
extern TensorDecDef *tensordec_find (const gchar *name);


G_END_DECLS
#endif /* __GST_TENSORDEC_H__ */
