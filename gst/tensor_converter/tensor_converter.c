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
 * @see		https://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */


/**
 *  @mainpage nnstreamer
 *  @section  intro         Introduction
 *  - Introduction      :   Neural Network Streamer for AI Projects
 *  @section   Program      Program Name
 *  - Program Name      :   nnstreamer
 *  - Program Details   :   It provides a neural network framework connectivities (e.g., tensorflow, caffe) for gstreamer streams.
 *    Efficient Streaming for AI Projects: Neural network models wanted to use efficient and flexible streaming management as well.
 *    Intelligent Media Filters!: Use a neural network model as a media filter / converter.
 *    Composite Models!: Allow to use multiple neural network models in a single stream instance.
 *    Multi Model Intelligence!: Allow to use multiple sources for neural network models.
 *  @section  INOUTPUT      Input/output data
 *  - INPUT             :   None
 *  - OUTPUT            :   None
 *  @section  CREATEINFO    Code information
 *  - Initial date      :   2018/06/14
 *  - Version           :   0.1
 */

/**
 * SECTION:element-tensor_converter
 *
 * A filter that converts media stream to tensor stream for NN frameworks.
 * The output is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! tensor_converter ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "tensor_converter.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

#define silent_debug(...) \
    debug_print (DBG, __VA_ARGS__)

#define silent_debug_caps(caps,msg) if (DBG) { \
  if (caps) { \
    GstStructure *caps_s; \
    gchar *caps_s_string; \
    guint caps_size, caps_idx; \
    caps_size = gst_caps_get_size (caps);\
    for (caps_idx = 0; caps_idx < caps_size; caps_idx++) { \
      caps_s = gst_caps_get_structure (caps, caps_idx); \
      caps_s_string = gst_structure_to_string (caps_s); \
      debug_print (TRUE, msg " = %s\n", caps_s_string); \
      g_free (caps_s_string); \
    } \
  } \
}

GST_DEBUG_CATEGORY_STATIC (gst_tensor_converter_debug);
#define GST_CAT_DEFAULT gst_tensor_converter_debug

/** properties */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_FORCE_MEMCPY,
  PROP_FRAMES_PER_BUFFER
};

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Disable in-place mode and do memcpy.
 */
#define DEFAULT_FORCE_MEMCPY FALSE

/**
 * @brief Samples per buffer to set tensor metadata.
 * (0 means sample rate for audio stream)
 */
#define DEFAULT_FRAMES_PER_BUFFER 0

#define SINK_CAPS \
    GST_STATIC_CAPS (GST_TENSOR_VIDEO_CAPS_STR "; " GST_TENSOR_AUDIO_CAPS_STR)

#define SRC_CAPS \
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)

/**
 * @brief The capabilities of the inputs
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    SINK_CAPS);

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    SRC_CAPS);

#define gst_tensor_converter_parent_class parent_class
G_DEFINE_TYPE (GstTensorConverter, gst_tensor_converter,
    GST_TYPE_BASE_TRANSFORM);

/** GObject vmethod implementations */
static void gst_tensor_converter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_converter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/** GstBaseTransform vmethod implementations */
static GstFlowReturn gst_tensor_converter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstFlowReturn gst_tensor_converter_transform_ip (GstBaseTransform *
    trans, GstBuffer * buf);
static GstCaps *gst_tensor_converter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensor_converter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_converter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_tensor_converter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);

/**
 * @brief Check tensor config is consistent
 * @param self "this" pointer to check consistency
 * @param config newly configured tensor metadata
 */
static gboolean
gst_tensor_converter_check_consistency (GstTensorConverter * self,
    const GstTensorConfig * config)
{
  g_return_val_if_fail (self != NULL, FALSE);
  g_return_val_if_fail (config != NULL, FALSE);

  if (self->tensor_configured) {
    return gst_tensor_config_is_same (&self->tensor_config, config);
  }

  /** not configured yet */
  return FALSE;
}

/**
 * @brief Parse structure and return tensor caps
 * @param self "this" pointer
 * @param structure structure to be interpreted
 */
static GstCaps *
gst_tensor_converter_caps_from_structure (GstTensorConverter * self,
    const GstStructure * structure)
{
  GstTensorConfig config;

  if (!gst_tensor_config_from_structure (&config, structure)) {
    return NULL;
  }

  /**
   * @todo How can we get the frames per buffer?
   * It depends on the tensorflow model, so added property frames-per-buffer.
   * If frames-per-buffer is 0, do nothing. (default sample rate)
   */
  if (config.tensor_media_type == _NNS_AUDIO) {
    if (self->frames_per_buffer > 0) {
      config.dimension[1] = self->frames_per_buffer;
    }
  }

  return gst_tensor_caps_from_config (&config);
}

/**
 * @brief Configure tensor metadata for video (internal static function)
 * @param self "this" pointer to be configured.
 * @param caps the sink cap.
 * @return FALSE if error. TRUE if ok.
 */
static gboolean
gst_tensor_converter_configure_for_video (GstTensorConverter * self,
    const GstCaps * caps)
{
  GstVideoInfo info;
  GstTensorConfig config;
  GstTensorVideoInfo v_info;

  gst_video_info_init (&info);
  if (!gst_video_info_from_caps (&info, caps)) {
    err_print ("Failed to get video info from caps.\n");
    return FALSE;
  }

  v_info.format = GST_VIDEO_INFO_FORMAT (&info);
  v_info.w = GST_VIDEO_INFO_WIDTH (&info);
  v_info.h = GST_VIDEO_INFO_HEIGHT (&info);
  v_info.fn = GST_VIDEO_INFO_FPS_N (&info);
  v_info.fd = GST_VIDEO_INFO_FPS_D (&info);

  silent_debug ("video info : format[%d] width[%d] height[%d] fps[%d/%d]",
      v_info.format, v_info.w, v_info.h, v_info.fn, v_info.fd);

  if (!gst_tensor_config_from_video_info (&config, &v_info)) {
    /** unsupported format */
    return FALSE;
  }

  if (self->tensor_configured &&
      !gst_tensor_converter_check_consistency (self, &config)) {
    /** mismatched to old metadata */
    return FALSE;
  }

  /**
   * Emit Warning if RSTRIDE = RU4 (3BPP) && Width % 4 > 0
   * @todo: Add more conditions!
   */
  if (gst_tensor_video_stride_padding_per_row (v_info.format, v_info.w)) {
    self->removePadding = TRUE;
  }

  self->tensor_config = config;
  self->in_info.video = info;
  return TRUE;
}

/**
 * @brief Configure tensor metadata for audio (internal static function)
 * @param self "this" pointer to be configured.
 * @param caps the sink cap.
 * @return FALSE if error. TRUE if ok.
 */
static gboolean
gst_tensor_converter_configure_for_audio (GstTensorConverter * self,
    const GstCaps * caps)
{
  GstAudioInfo info;
  GstTensorConfig config;
  GstTensorAudioInfo a_info;

  gst_audio_info_init (&info);
  if (!gst_audio_info_from_caps (&info, caps)) {
    err_print ("Failed to get audio info from caps.\n");
    return FALSE;
  }

  a_info.format = GST_AUDIO_INFO_FORMAT (&info);
  a_info.ch = GST_AUDIO_INFO_CHANNELS (&info);
  a_info.rate = GST_AUDIO_INFO_RATE (&info);
  a_info.frames =
      (self->frames_per_buffer > 0) ? self->frames_per_buffer : a_info.rate;

  silent_debug ("audio info : format[%d] channel[%d] rate[%d] bpf[%d]",
      a_info.format, a_info.ch, a_info.rate, GST_AUDIO_INFO_BPF (&info));

  if (!gst_tensor_config_from_audio_info (&config, &a_info)) {
    /** unsupported format */
    return FALSE;
  }

  if (self->tensor_configured &&
      !gst_tensor_converter_check_consistency (self, &config)) {
    /** mismatched to old metadata */
    return FALSE;
  }

  self->tensor_config = config;
  self->in_info.audio = info;
  return TRUE;
}

/**
 * @brief Configure tensor metadata from sink caps (internal static function)
 * @param self "this" pointer to be configured.
 * @param caps the sink cap.
 * @return FALSE if error. TRUE if ok.
 */
static gboolean
gst_tensor_converter_configure_tensor (GstTensorConverter * self,
    const GstCaps * caps)
{
  media_type m_type;

  m_type = gst_tensor_media_type_from_caps (caps);

  /** @todo Support other types */
  switch (m_type) {
    case _NNS_VIDEO:
      if (!gst_tensor_converter_configure_for_video (self, caps)) {
        return FALSE;
      }
      break;

    case _NNS_AUDIO:
      if (!gst_tensor_converter_configure_for_audio (self, caps)) {
        return FALSE;
      }
      break;

    default:
      err_print ("Unsupported type %d\n", m_type);
      return FALSE;
  }

  self->tensor_configured = TRUE;
  return TRUE;
}

/**
 * @brief initialize the tensor_converter's class
 */
static void
gst_tensor_converter_class_init (GstTensorConverterClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_converter_set_property;
  gobject_class->get_property = gst_tensor_converter_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FORCE_MEMCPY,
      g_param_spec_boolean ("force-memcpy", "Force Memcpy",
          "Disable in-place mode and do memcpy ?", DEFAULT_FORCE_MEMCPY,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FRAMES_PER_BUFFER,
      g_param_spec_uint ("frames-per-buffer", "Frames per buffer",
          "Sample count per buffer", 0, G_MAXUINT32, DEFAULT_FRAMES_PER_BUFFER,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_details_simple (gstelement_class,
      "Tensor_Converter",
      "Convert media stream to tensor stream",
      "Converts audio or video stream to tensor stream of C-Array for neural network framework filters",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
  /** Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /** Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensor_converter_transform);
  trans_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_tensor_converter_transform_ip);

  /** Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_converter_transform_caps);
  trans_class->fixate_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_converter_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_converter_set_caps);

  /** Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensor_converter_transform_size);
}

/**
 * @brief initialize the new element (G_DEFINE_TYPE requires this)
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_converter_init (GstTensorConverter * self)
{
  self->silent = DEFAULT_SILENT;
  self->tensor_configured = FALSE;
  self->negotiated = FALSE;
  self->removePadding = FALSE;
  self->disableInPlace = DEFAULT_FORCE_MEMCPY;
  self->frames_per_buffer = DEFAULT_FRAMES_PER_BUFFER;
  gst_tensor_config_init (&self->tensor_config);
}

/**
 * @brief Set property (GObject vmethod)
 */
static void
gst_tensor_converter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorConverter *self = GST_TENSOR_CONVERTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    case PROP_FORCE_MEMCPY:
      self->disableInPlace = g_value_get_boolean (value);
      break;
    case PROP_FRAMES_PER_BUFFER:
      self->frames_per_buffer = g_value_get_uint (value);
      silent_debug ("Set frames per buffer = %d", self->frames_per_buffer);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Get property (GObject vmethod)
 */
static void
gst_tensor_converter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorConverter *self = GST_TENSOR_CONVERTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_FORCE_MEMCPY:
      g_value_set_boolean (value, self->disableInPlace);
      break;
    case PROP_FRAMES_PER_BUFFER:
      g_value_set_uint (value, self->frames_per_buffer);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief copies sink pad buffer to src pad buffer (internal static function)
 * @param self "this" pointer
 * @param inbuf sink pad buffer
 * @param outbuf src pad buffer
 * @return GST_FLOW_OK if ok. other values represents error
 */
static GstFlowReturn
gst_tensor_converter_copy_buffer (GstTensorConverter * self,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstMapInfo src_info, dest_info;
  unsigned char *srcptr, *destptr;
  GstTensorConfig *config;
  gsize block_size;

  g_assert (self->tensor_configured);

  config = &self->tensor_config;
  block_size = config->frame_size;
  g_assert (gst_buffer_get_size (outbuf) >= block_size);

  g_assert (gst_buffer_map (inbuf, &src_info, GST_MAP_READ));
  g_assert (gst_buffer_map (outbuf, &dest_info, GST_MAP_WRITE));

  srcptr = src_info.data;
  destptr = dest_info.data;

  switch (config->tensor_media_type) {
    case _NNS_VIDEO:
      if (self->removePadding == TRUE) {
        int d0, d1;
        unsigned int src_idx = 0, dest_idx = 0;
        size_t size, offset;

        size = offset = config->dimension[0] * config->dimension[1];

        g_assert (offset % 4);
        if (offset % 4)
          offset += 4 - (offset % 4);
        /** Refer: https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html */

        for (d0 = 0; d0 < config->dimension[3]; d0++) { /** Supposed to be 0 only */
          g_assert (d0 == 0);
          for (d1 = 0; d1 < config->dimension[2]; d1++) { /** Height */
            memcpy (destptr + dest_idx, srcptr + src_idx, size);
            dest_idx += size;
            src_idx += offset;
          }
        }

        err_print
            ("\n\n\nYOUR STREAM CONFIGURATION INCURS PERFORMANCE DETERIORATION! (1)\nPlease use 4 x n as image width for inputs.\n\n\n");
        goto done;
      }
      break;

    default:
      break;
  }

  memcpy (destptr, srcptr, block_size);

done:
  gst_buffer_unmap (inbuf, &src_info);
  gst_buffer_unmap (outbuf, &dest_info);
  return GST_FLOW_OK;
}

/**
 * @brief non-ip transform. required vmethod for BaseTransform class.
 */
static GstFlowReturn
gst_tensor_converter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensorConverter *self;
  GstTensorConfig *config;
  GstFlowReturn res;

  self = GST_TENSOR_CONVERTER_CAST (trans);

  if (G_UNLIKELY (!self->negotiated))
    goto unknown_format;
  if (G_UNLIKELY (!self->tensor_configured))
    goto unknown_tensor;

  config = &self->tensor_config;

  switch (config->tensor_media_type) {
    case _NNS_VIDEO:
    case _NNS_AUDIO:
      res = gst_tensor_converter_copy_buffer (self, inbuf, outbuf);
      break;

    case _NNS_STRING:
    default:
      err_print ("Unsupported Media Type (%d)\n", config->tensor_media_type);
      goto unknown_type;
  }

  return res;

unknown_format:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_tensor:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL),
      ("unknown format for tensor"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_type:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL),
      ("not implemented type of media"));
  return GST_FLOW_NOT_SUPPORTED;
}

/**
 * @brief in-place transform. required vmethod for BaseTransform class.
 */
static GstFlowReturn
gst_tensor_converter_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstTensorConverter *self;
  GstTensorConfig *config;

  self = GST_TENSOR_CONVERTER_CAST (trans);

  if (G_UNLIKELY (!self->negotiated))
    goto unknown_format;
  if (G_UNLIKELY (!self->tensor_configured))
    goto unknown_tensor;

  config = &self->tensor_config;

  switch (config->tensor_media_type) {
    case _NNS_VIDEO:
      if (self->removePadding == TRUE) {
        /** Remove zero-padding between rows */
        GstMapInfo info;
        unsigned char *ptr;
        unsigned int row, d0;
        unsigned int src_idx = 0, dest_idx = 0;
        size_t size, offset;

        size = offset = config->dimension[0] * config->dimension[1];

        g_assert (offset % 4);
        if (offset % 4)
          offset += 4 - (offset % 4);
        /** Refer: https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html */

        gst_buffer_map (buf, &info, GST_MAP_READWRITE);
        ptr = info.data;

        for (d0 = 0; d0 < config->dimension[3]; d0++) { /** Supposed to be 0 only */
          g_assert (d0 == 0);
          for (row = 0; row < config->dimension[2]; row++) { /** Height */
            if (dest_idx != src_idx)
              memmove (ptr + dest_idx, ptr + src_idx, size);
            dest_idx += size;
            src_idx += offset;
          }
        }
        /** @todo: Remove the clutter (reduce the size?) after memcpy. (Check if that's really helpful, first) */
        gst_buffer_unmap (buf, &info);

        err_print
            ("\n\n\nYOUR STREAM CONFIGURATION INCURS PERFORMANCE DETERIORATION! (2)\nPlease use 4 x n as image width for inputs.\n\n\n");
      }
      break;

    case _NNS_AUDIO:
      break;

    case _NNS_STRING:
    default:
      err_print ("Unsupported Media Type (%d)\n", config->tensor_media_type);
      goto unknown_type;
  }

  /** DO NOTHING. THIS WORKS AS A PASSTHROUGH. We just remove metadata from video */
  return GST_FLOW_OK;

unknown_format:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_tensor:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL),
      ("unknown format for tensor"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_type:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL),
      ("not implemented type of media"));
  return GST_FLOW_NOT_SUPPORTED;
}

/**
 * @brief configure srcpad cap from "proposed" cap. (required vmethod for BaseTransform)
 *
 * @param trans ("this" pointer)
 * @param direction (why do we need this?)
 * @param caps sinkpad cap
 * @param filter this element's cap (don't know specifically.)
 */
static GstCaps *
gst_tensor_converter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstTensorConverter *self;
  GstCaps *result;

  self = GST_TENSOR_CONVERTER_CAST (trans);

  silent_debug_caps (caps, "from");
  silent_debug_caps (filter, "filter");

  if (direction == GST_PAD_SINK) {
    GstStructure *s = gst_caps_get_structure (caps, 0);
    result = gst_tensor_converter_caps_from_structure (self, s);
  } else if (direction == GST_PAD_SRC) {
    GstStaticCaps raw_sink_caps = SINK_CAPS;
    result = gst_static_caps_get (&raw_sink_caps);
  } else {
    silent_debug ("Direction = %d\n", direction);
    g_assert (0);
    return NULL;
  }

  if (filter) {
    GstCaps *intersection;

    intersection =
        gst_caps_intersect_full (filter, result, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (result);
    result = intersection;
  }

  silent_debug_caps (result, "to");

  GST_DEBUG_OBJECT (trans, "Direction[%d] transformed %" GST_PTR_FORMAT
      " into %" GST_PTR_FORMAT, direction, caps, result);
  return result;
}

/**
 * @brief fixate caps. required vmethod of BaseTransform
 */
static GstCaps *
gst_tensor_converter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstTensorConverter *self;
  GstCaps *result;

  self = GST_TENSOR_CONVERTER_CAST (trans);

  silent_debug_caps (caps, "from caps");
  silent_debug_caps (othercaps, "from othercaps");

  GST_DEBUG_OBJECT (trans, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, caps);

  if (direction == GST_PAD_SINK) {
    GstCaps *supposed;

    if (gst_tensor_converter_configure_tensor (self, caps)) {
      supposed = gst_tensor_caps_from_config (&self->tensor_config);
    } else {
      /**
       * Failed to get media info (caps is not fixed yet)
       * Parse caps from structure.
       */
      GstStructure *s = gst_caps_get_structure (caps, 0);
      supposed = gst_tensor_converter_caps_from_structure (self, s);
    }

    result = gst_caps_intersect (othercaps, supposed);
    gst_caps_unref (supposed);
  } else {
    result = gst_caps_copy (othercaps);
  }

  if (gst_caps_is_empty (result)) {
    gst_caps_unref (result);
    result = othercaps;
  } else {
    gst_caps_unref (othercaps);
  }

  GST_DEBUG_OBJECT (trans, "now fixating %" GST_PTR_FORMAT, result);

  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

  if (direction == GST_PAD_SINK) {
    if (gst_caps_is_subset (caps, result)) {
      gst_caps_replace (&result, caps);
    }
  }
  return result;
}

/**
 * @brief set caps. required vmethod of BaseTransform
 */
static gboolean
gst_tensor_converter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  /**
   * This is notifier of cap changes for subclass.
   * However, we do not have subclass (This is the concrete class)
   */
  GstTensorConverter *self;

  self = GST_TENSOR_CONVERTER_CAST (trans);

  silent_debug_caps (incaps, "from incaps");
  silent_debug_caps (outcaps, "from outcaps");

  GST_DEBUG_OBJECT (trans, "converting from  %" GST_PTR_FORMAT
      " to %" GST_PTR_FORMAT, incaps, outcaps);

  gst_base_transform_set_in_place (trans, !self->disableInPlace);

  /** compare and verify outcaps */
  if (gst_tensor_converter_configure_tensor (self, incaps)) {
    GstTensorConfig config;
    GstStructure *s = gst_caps_get_structure (outcaps, 0);

    if (gst_tensor_config_from_structure (&config, s) &&
        gst_tensor_converter_check_consistency (self, &config)) {
      self->negotiated = TRUE;
    }
  }

  /** now return true if negotiated */
  return self->negotiated;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. optional vmethod of BaseTransform
 *
 * We cannot directly get the value from size value, we need to review the pad-caps.
 * This is called when non-ip mode is used.
 */
static gboolean
gst_tensor_converter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  GstTensorConverter *self;

  self = GST_TENSOR_CONVERTER_CAST (trans);

  g_assert (self->tensor_configured);
  *othersize = self->tensor_config.frame_size;

  return TRUE;
}

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
gst_tensor_converter_plugin_init (GstPlugin * plugin)
{
  /**
   * debug category for fltering log messages
   *
   * exchange the string 'Template tensor_converter' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_converter_debug, "tensor_converter",
      0, "Template tensor_converter");

  return gst_element_register (plugin, "tensor_converter",
      GST_RANK_NONE, GST_TYPE_TENSOR_CONVERTER);
}

/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensor_converter"
#endif

/**
 * gstreamer looks for this structure to register tensor_converters
 *
 * exchange the string 'Template tensor_converter' with your tensor_converter description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_converter,
    "tensor_converter",
    gst_tensor_converter_plugin_init,
    VERSION, "LGPL", "GStreamer", "http://gstreamer.net/");
