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
typedef struct _GstTensorDec GstTensorDec;
typedef struct _GstTensorDecClass GstTensorDecClass;
typedef struct _TensorDecDef TensorDecDef;


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
 * @brief Tensor Decoder Output type.
 */
typedef enum
{
  OUTPUT_VIDEO,
  OUTPUT_AUDIO,
  OUTPUT_TEXT,
  OUTPUT_UNKNOWN
} GstDecMediaType;

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
  char *modename;
      /**< Unique decoder name. GST users choose decoders with mode="modename". */
  GstDecMediaType type;
      /**< Output media type. VIDEO/AUDIO/TEXT are supported */
  int (*init) (void **private_data);
      /**< Object initialization for the decoder */
  void (*exit) (void **private_data);
      /**< Object destruction for the decoder */
  int (*setOption) (void **private_data, int opNum, const char *param);
      /**< Process with the given options. It can be called repeatedly */
  GstCaps *(*getOutCaps) (void **private_data, const GstTensorsConfig *config);
      /**< The caller should unref the returned GstCaps
        * Current implementation supports single-tensor only.
        * @todo WIP: support multi-tensor for input!!!
        */
  GstFlowReturn (*decode) (void **private_data, const GstTensorsConfig *config,
      const GstTensorMemory *input, GstBuffer *outbuf);
      /**< outbuf must be allocated but empty (gst_buffer_get_size (outbuf) == 0).
        * Note that we support single-tensor (other/tensor) only!
        * @todo WIP: support multi-tensor for input!!!
        */
  size_t (*getTransformSize) (void **private_data, const GstTensorsConfig *config,
      GstCaps *caps, size_t size, GstCaps *othercaps,
      GstPadDirection direction);
      /**< EXPERIMENTAL! @todo We are not ready to use this. This should be NULL or return 0 */
};

/* extern functions for subplugin management, exist in tensor_decoder.c */
/**
 * @brief decoder's subplugins should call this function to register
 * @param[in] decoder The decoder subplugin instance
 */
extern gboolean tensordec_probe (TensorDecDef * decoder);
/**
 * @brief decoder's subplugin may call this to unregister
 * @param[in] name the name of decoder (modename)
 */
extern void tensordec_exit (const gchar * name);
G_END_DECLS
#endif /* __GST_TENSORDEC_H__ */
