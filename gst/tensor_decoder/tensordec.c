/**
 * GStreamer
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
 * @file	tensordec.c
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert tensors (as a filter for other general neural network filters) to other media types
 * @see		https://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

/**
 * SECTION:element-tensordec
 *
 * A filter that converts tensor stream for NN frameworks to media stream.
 * The input is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesink ! tensor_decoder ! fakesrc silent=TRUE
 * ]|
 * </refsect2>
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <stdio.h>
#include <glib.h>
#include <string.h>
#include <stdlib.h>
#include "tensordec.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

#define silent_debug(...) \
    debug_print (DBG, __VA_ARGS__)

#define silent_debug_caps(caps,msg) do {\
  if (DBG) { \
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
  } \
} while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensordec_debug);
#define GST_CAT_DEFAULT gst_tensordec_debug

/**
 * @brief Output type.
 * @todo Change output type (eg, image box, label)
 */
enum
{
  OUTPUT_VIDEO,
  OUTPUT_AUDIO,
  OUTPUT_TEXT,
  OUTPUT_UNKNOWN
};

/**
 * @brief Properties.
 */
enum
{
  PROP_0,
  PROP_OUTPUT_TYPE,
  PROP_SILENT,
  PROP_MODE,
  PROP_MODE_OPTION1
};

/**
 * @brief Default output type.
 */
#define DEFAULT_OUTPUT_TYPE OUTPUT_VIDEO

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief The capabilities of the inputs
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT));

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY"));

#define gst_tensordec_parent_class parent_class
G_DEFINE_TYPE (GstTensorDec, gst_tensordec, GST_TYPE_BASE_TRANSFORM);

/** GObject vmethod implementations */
static void gst_tensordec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensordec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/** GstBaseTransform vmethod implementations */
static GstFlowReturn gst_tensordec_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstFlowReturn gst_tensordec_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);
static GstCaps *gst_tensordec_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensordec_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensordec_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_tensordec_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);

/**
 * @brief initialize data in tensor decoder image labeling info structure.
 */
static void
gst_tensordec_image_labeling_init (Mode_image_labeling * mode_image_label)
{
  mode_image_label->label_path = NULL;
  mode_image_label->labels = NULL;
  mode_image_label->total_labels = 0;
}

/**
 * @brief set label info data for Tensor decoder.
 */
static gboolean
gst_set_mode_image_label_info (GstTensorDec * self)
{
  FILE *fp;

  if ((fp = fopen (self->tensordec_image_label.label_path, "r")) != NULL) {
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    gchar *label;

    while ((read = getline (&line, &len, fp)) != -1) {
      label = g_strdup ((gchar *) line);
      self->tensordec_image_label.labels =
          g_list_append (self->tensordec_image_label.labels, label);
    }

    if (line) {
      free (line);
    }

    fclose (fp);
  } else {
    err_print ("cannot find label file in tensor decoder");
    return FALSE;
  }

  self->tensordec_image_label.total_labels =
      g_list_length (self->tensordec_image_label.labels);
  err_print ("finished to load labels");
  return TRUE;
}

/**
 * @brief Get video caps from tensor config
 * @param self "this" pointer
 * @param config tensor config info
 * @return caps for given config
 */
static GstCaps *
gst_tensordec_video_caps_from_config (GstTensorDec * self,
    const GstTensorConfig * config)
{
  GstVideoFormat format;
  gint width, height, fn, fd;
  GstCaps *caps;

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_caps_from_string (GST_TENSOR_VIDEO_CAPS_STR);

  switch (config->info.dimension[0]) {
    case 1:
      format = GST_VIDEO_FORMAT_GRAY8;
      break;
    case 3:
      format = GST_VIDEO_FORMAT_RGB;
      break;
    case 4:
      format = GST_VIDEO_FORMAT_BGRx;
      break;
    default:
      format = GST_VIDEO_FORMAT_UNKNOWN;
      break;
  }

  width = config->info.dimension[1];
  height = config->info.dimension[2];
  fn = config->rate_n;
  fd = config->rate_d;

  if (format != GST_VIDEO_FORMAT_UNKNOWN) {
    const gchar *format_string = gst_video_format_to_string (format);
    gst_caps_set_simple (caps, "format", G_TYPE_STRING, format_string, NULL);
  }

  if (width > 0) {
    gst_caps_set_simple (caps, "width", G_TYPE_INT, width, NULL);
  }

  if (height > 0) {
    gst_caps_set_simple (caps, "height", G_TYPE_INT, height, NULL);
  }

  if (fn > 0 && fd > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, fn, fd, NULL);
  }

  return gst_caps_simplify (caps);
}

/**
 * @brief Get audio caps from tensor config
 * @param self "this" pointer
 * @param config tensor config info
 * @return caps for given config
 */
static GstCaps *
gst_tensordec_audio_caps_from_config (GstTensorDec * self,
    const GstTensorConfig * config)
{
  GstAudioFormat format;
  gint ch, rate;
  GstCaps *caps;

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_caps_from_string (GST_TENSOR_AUDIO_CAPS_STR);

  switch (config->info.type) {
    case _NNS_INT8:
      format = GST_AUDIO_FORMAT_S8;
      break;
    case _NNS_UINT8:
      format = GST_AUDIO_FORMAT_U8;
      break;
    case _NNS_INT16:
      format = GST_AUDIO_FORMAT_S16;
      break;
    case _NNS_UINT16:
      format = GST_AUDIO_FORMAT_U16;
      break;
    default:
      format = GST_AUDIO_FORMAT_UNKNOWN;
      break;
  }

  ch = config->info.dimension[0];
  rate = config->rate_n;

  if (format != GST_AUDIO_FORMAT_UNKNOWN) {
    const gchar *format_string = gst_audio_format_to_string (format);
    gst_caps_set_simple (caps, "format", G_TYPE_STRING, format_string, NULL);
  }

  if (ch > 0) {
    gst_caps_set_simple (caps, "channels", G_TYPE_INT, ch, NULL);
  }

  if (rate > 0) {
    gst_caps_set_simple (caps, "rate", G_TYPE_INT, rate, NULL);
  }

  return gst_caps_simplify (caps);
}

/**
 * @brief Get text caps from tensor config
 * @param self "this" pointer
 * @param config tensor config info
 * @return caps for given config
 */
static GstCaps *
gst_tensordec_text_caps_from_config (GstTensorDec * self,
    const GstTensorConfig * config)
{
  g_return_val_if_fail (config != NULL, NULL);

  /**
   * Set text format. Supposed utf8 if type is int8.
   */
  g_return_val_if_fail (config->info.type == _NNS_INT8, NULL);

  return gst_caps_from_string (GST_TENSOR_TEXT_CAPS_STR);
}

/**
 * @brief Get media caps from tensor config
 * @param self "this" pointer
 * @param config tensor config info
 * @return caps for media type
 */
static GstCaps *
gst_tensordec_media_caps_from_config (GstTensorDec * self,
    const GstTensorConfig * config)
{
  GstCaps *caps = NULL;

  g_return_val_if_fail (config != NULL, NULL);

  switch (self->output_type) {
    case OUTPUT_VIDEO:
      caps = gst_tensordec_video_caps_from_config (self, config);
      break;
    case OUTPUT_AUDIO:
      caps = gst_tensordec_audio_caps_from_config (self, config);
      break;
    case OUTPUT_TEXT:
      caps = gst_tensordec_text_caps_from_config (self, config);
      break;
    default:
      err_print ("Unsupported type %d\n", self->output_type);
      break;
  }

  return caps;
}

/**
 * @brief Parse structure and return media caps
 * @param self "this" pointer
 * @param structure structure to be interpreted
 */
static GstCaps *
gst_tensordec_media_caps_from_structure (GstTensorDec * self,
    const GstStructure * structure)
{
  GstTensorConfig config;
  GstCaps *result = NULL;

  if (gst_tensor_config_from_structure (&config, structure)) {
    result = gst_tensordec_media_caps_from_config (self, &config);
  }

  if (result == NULL) {
    /** raw caps for supported media types */
    result = gst_caps_from_string (GST_TENSOR_MEDIA_CAPS_STR);
  }

  return result;
}

/**
 * @brief Check tensor config is consistent
 * @param self "this" pointer to check consistency
 * @param t_info newly configured tensor metadata
 */
static gboolean
gst_tensordec_check_consistency (GstTensorDec * self, GstTensorConfig * config)
{
  g_return_val_if_fail (self != NULL, FALSE);
  g_return_val_if_fail (config != NULL, FALSE);

  if (self->configured) {
    return gst_tensor_config_is_equal (&self->tensor_config, config);
  }

  /** not configured yet */
  return FALSE;
}

/**
 * @brief initialize the tensordec's class
 */
static void
gst_tensordec_class_init (GstTensorDecClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensordec_set_property;
  gobject_class->get_property = gst_tensordec_get_property;

  g_object_class_install_property (gobject_class, PROP_OUTPUT_TYPE,
      g_param_spec_uint ("output-type", "Output type",
          "Output type from the plugin", 0, G_MAXUINT,
          DEFAULT_OUTPUT_TYPE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_details_simple (gstelement_class,
      "TensorDecoder",
      "Converter/Tensor",
      "Converts tensor stream of C-Array for neural network framework filters to audio or video stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));

  /** Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /** Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensordec_transform);
  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_tensordec_transform_ip);

  /** Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensordec_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensordec_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensordec_set_caps);

  /** Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensordec_transform_size);
}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensordec_init (GstTensorDec * self)
{
  self->silent = DEFAULT_SILENT;
  self->configured = FALSE;
  self->negotiated = FALSE;
  self->add_padding = FALSE;
  self->output_type = OUTPUT_VIDEO;
  self->mode = Mode[0];
  gst_tensor_config_init (&self->tensor_config);
  gst_tensordec_image_labeling_init (&self->tensordec_image_label);
}

/**
 * @brief Set property (GObject vmethod)
 */
static void
gst_tensordec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorDec *self = GST_TENSORDEC (object);

  switch (prop_id) {
    case PROP_OUTPUT_TYPE:
      self->output_type = g_value_get_uint (value);
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    case PROP_MODE:
      self->mode = g_value_dup_string (value);
      break;
    case PROP_MODE_OPTION1:
      if (g_strcmp0 (self->mode, "image_labeling") == 0) {
        self->tensordec_image_label.label_path = g_value_dup_string (value);
        gst_set_mode_image_label_info (self);
      }
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
gst_tensordec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorDec *self = GST_TENSORDEC (object);

  switch (prop_id) {
    case PROP_OUTPUT_TYPE:
      g_value_set_uint (value, self->output_type);
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_MODE:
      g_value_set_string (value, self->mode);
      break;
    case PROP_MODE_OPTION1:
      g_value_set_string (value, self->tensordec_image_label.label_path);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Configure tensor metadata from sink caps
 */
static gboolean
gst_tensordec_configure (GstTensorDec * self, const GstCaps * caps)
{
  GstStructure *structure;
  GstTensorConfig config;

  /** This caps is coming from tensor */
  structure = gst_caps_get_structure (caps, 0);

  if (!gst_tensor_config_from_structure (&config, structure)) {
    err_print ("Cannot configure tensor from structure");
    return FALSE;
  }

  if (!gst_tensor_config_validate (&config)) {
    err_print ("Not configured yet");
    return FALSE;
  }

  if (self->configured && !gst_tensordec_check_consistency (self, &config)) {
    err_print ("Mismatched to old metadata");
    return FALSE;
  }

  switch (self->output_type) {
    case OUTPUT_VIDEO:
    {
      GstVideoFormat format;
      gint width;

      switch (config.info.dimension[0]) {
        case 1:
          format = GST_VIDEO_FORMAT_GRAY8;
          break;
        case 3:
          format = GST_VIDEO_FORMAT_RGB;
          break;
        case 4:
          format = GST_VIDEO_FORMAT_BGRx;
          break;
        default:
          format = GST_VIDEO_FORMAT_UNKNOWN;
          break;
      }
      width = config.info.dimension[1];

      if (gst_tensor_video_stride_padding_per_row (format, width)) {
        self->add_padding = TRUE;
      }

      break;
    }
    case OUTPUT_AUDIO:
    case OUTPUT_TEXT:
      break;
    default:
      err_print ("Unsupported type %d\n", self->output_type);
      return FALSE;
  }

  self->tensor_config = config;
  self->configured = TRUE;
  return TRUE;
}

/**
 * @brief copies sink pad buffer to src pad buffer (internal static function)
 * @param self "this" pointer
 * @param inbuf sink pad buffer
 * @param outbuf src pad buffer
 * @return GST_FLOW_OK if ok. other values represents error
 */
static GstFlowReturn
gst_tensordec_copy_buffer (GstTensorDec * self,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstMapInfo inInfo, outInfo;
  uint8_t *inptr, *outptr;
  GstTensorConfig *config;
  unsigned int row, d0;
  unsigned int dest_idx = 0, src_idx = 0;
  size_t size, offset, size_out;

  g_assert (self->configured);

  config = &self->tensor_config;

  /** flag add_padding only for video */
  g_assert (self->add_padding);
  g_assert (self->output_type == OUTPUT_VIDEO);

  size = offset = config->info.dimension[0] * config->info.dimension[1];

  if (offset % 4)
    offset += 4 - (offset % 4);

  size_out = offset * config->info.dimension[2] * config->info.dimension[3];

  if (gst_buffer_get_size (outbuf) < size_out) {
    gst_buffer_set_size (outbuf, size_out);
  }

  gst_buffer_map (inbuf, &inInfo, GST_MAP_READ);
  gst_buffer_map (outbuf, &outInfo, GST_MAP_WRITE);

  inptr = inInfo.data;
  outptr = outInfo.data;

  for (d0 = 0; d0 < config->info.dimension[3]; d0++) {
    g_assert (d0 == 0);
    for (row = 0; row < config->info.dimension[2]; row++) {
      memcpy (outptr + dest_idx, inptr + src_idx, size);
      dest_idx += offset;
      src_idx += size;
    }
  }

  gst_buffer_unmap (inbuf, &inInfo);
  gst_buffer_unmap (outbuf, &outInfo);

  return GST_FLOW_OK;
}

/**
 * @brief update top label index by given tensor data 
 * @param self "this" pointer
 * @param scores given tensor data
 * @param len length of valid given tensor data
 */
static gint
gst_tensordec_update_top_label_index (GstTensorDec * self,
    guint8 * scores, guint len)
{
  gint i;
  gint ret;
  gint index = -1;
  guint8 max_score = 0;

  /** -1 if failed to get max score index */
  ret = -1;

  g_return_if_fail (scores != NULL);
  g_return_if_fail (len == self->tensordec_image_label.total_labels);

  for (i = 0; i < len; i++) {
    if (scores[i] > 0 && scores[i] > max_score) {
      index = i;
      max_score = scores[i];
    }
  }

  ret = index;
  return ret;
}

/**
 * @brief get image label text with given index 
 * @param self "this" pointer
 */
static gchar *
gst_get_image_label (GstTensorDec * self, gint label)
{
  guint length;
  guint check_label = label;
  g_return_val_if_fail (self != NULL, NULL);
  g_return_val_if_fail (self->tensordec_image_label.labels != NULL, NULL);

  length = g_list_length (self->tensordec_image_label.labels);
  g_return_val_if_fail (check_label >= 0 && check_label < length, NULL);

  return (gchar *) g_list_nth_data
      (self->tensordec_image_label.labels, check_label);
}

/**
 * @brief set output of tensor decoder that will send to src pad  
 * @param self "this" pointer
 * @param outbuf src pad buffer
 * @param label image label text that will copy to outbuf
 */
static void
gst_tensordec_label_set_output (GstTensorDec * self, GstBuffer * outbuf,
    gchar * label)
{
  guint len;
  GstMemory *out_mem;
  GstMapInfo out_info;

  g_assert (gst_buffer_get_size (outbuf) == 0);

  len = strlen (label);

  out_mem = gst_allocator_alloc (NULL, (len + 1), NULL);
  g_assert (out_mem != NULL);
  g_assert (gst_memory_map (out_mem, &out_info, GST_MAP_WRITE));

  strncpy ((char *) out_info.data, label, len);

  gst_buffer_append_memory (outbuf, out_mem);

  gst_memory_unmap (out_mem, &out_info);
}

/**
 * @brief get image label by incomming tensor 
 * @param self "this" pointer
 * @param inbuf sink pad buffer
 * @param outbuf src pad buffer
 * @return GST_FLOW_OK if ok. other values represents error
 */
static GstFlowReturn
gst_tensordec_get_label (GstTensorDec * self,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstMemory *mem;
  GstMapInfo info;
  gchar *image_label;
  guint i;
  gint max_label = -1;
  guint num_mems;

  num_mems = gst_buffer_n_memory (inbuf);
  for (i = 0; i < num_mems; i++) {
    mem = gst_buffer_peek_memory (inbuf, i);
    if (gst_memory_map (mem, &info, GST_MAP_READ)) {
      /** update label index with max score */
      max_label =
          gst_tensordec_update_top_label_index (self, info.data,
          (guint) info.size);
      gst_memory_unmap (mem, &info);
    }
  }
  image_label = gst_get_image_label (self, max_label);
  gst_tensordec_label_set_output (self, outbuf, image_label);
  return GST_FLOW_OK;
}

/**
 * @brief non-ip transform. required vmethod for BaseTransform class.
 */
static GstFlowReturn
gst_tensordec_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensorDec *self;
  GstFlowReturn res;

  self = GST_TENSORDEC_CAST (trans);

  if (G_UNLIKELY (!self->negotiated))
    goto unknown_tensor;
  if (G_UNLIKELY (!self->configured))
    goto unknown_format;

  switch (self->output_type) {
    case OUTPUT_VIDEO:
    case OUTPUT_AUDIO:
      res = gst_tensordec_copy_buffer (self, inbuf, outbuf);
      break;
    case OUTPUT_TEXT:
      if (g_strcmp0 (self->mode, "image_labeling") == 0) {
        res = gst_tensordec_get_label (self, inbuf, outbuf);
      } else {
        res = gst_tensordec_copy_buffer (self, inbuf, outbuf);
      }
      break;
    default:
      err_print ("Unsupported Media Type (%d)\n", self->output_type);
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
gst_tensordec_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstTensorDec *self;

  self = GST_TENSORDEC_CAST (trans);

  if (G_UNLIKELY (!self->negotiated))
    goto unknown_format;
  if (G_UNLIKELY (!self->configured))
    goto unknown_tensor;

  switch (self->output_type) {
    case OUTPUT_VIDEO:
      if (self->add_padding) {
        /**
         * @todo Do we need to add padding for x-raw here?
         */
      }
      break;
    case OUTPUT_AUDIO:
    case OUTPUT_TEXT:
      break;
    default:
      err_print ("Unsupported Media Type (%d)\n", self->output_type);
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
 * @brief configure tensor-srcpad cap from "proposed" cap.
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap
 * @filter this element's cap (don't know specifically.)
 */
static GstCaps *
gst_tensordec_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstTensorDec *self;
  GstCaps *result;

  self = GST_TENSORDEC_CAST (trans);

  silent_debug ("Direction = %d\n", direction);
  silent_debug_caps (caps, "from");
  silent_debug_caps (filter, "filter");

  /**
   * If direction is sink, check src. Depending on sink's format, we could choose video or audio.
   * Currently video/x-raw and audio/x-raw supported.
   */
  if (direction == GST_PAD_SINK) {
    /** caps from media */
    GstStructure *s = gst_caps_get_structure (caps, 0);
    result = gst_tensordec_media_caps_from_structure (self, s);
  } else if (direction == GST_PAD_SRC) {
    /** caps from tensor */
    result = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT);
  } else {
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
gst_tensordec_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstTensorDec *self;
  GstCaps *supposed;
  GstCaps *result;

  self = GST_TENSORDEC_CAST (trans);

  silent_debug_caps (caps, "from caps");
  silent_debug_caps (othercaps, "from othercaps");

  GST_DEBUG_OBJECT (trans, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, caps);

  if (gst_tensordec_configure (self, caps)) {
    supposed =
        gst_tensordec_media_caps_from_config (self, &self->tensor_config);
  } else {
    GstStructure *s = gst_caps_get_structure (caps, 0);
    supposed = gst_tensordec_media_caps_from_structure (self, s);
  }

  result = gst_caps_intersect (othercaps, supposed);
  gst_caps_unref (supposed);

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
gst_tensordec_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstTensorDec *self;

  self = GST_TENSORDEC_CAST (trans);

  silent_debug_caps (incaps, "from incaps");
  silent_debug_caps (outcaps, "from outcaps");

  /** compare and verify outcaps */
  if (gst_tensordec_configure (self, incaps)) {
    GstTensorConfig config;
    GstStructure *s = gst_caps_get_structure (outcaps, 0);

    if (gst_tensor_config_from_structure (&config, s) &&
        gst_tensordec_check_consistency (self, &config)) {
      self->negotiated = TRUE;
    }
  }

  gst_base_transform_set_in_place (trans, !self->add_padding);

  return self->negotiated;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. optional vmethod of BaseTransform
 *
 * This is called when non-ip mode is used.
 */
static gboolean
gst_tensordec_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  GstTensorDec *self;
  GstTensorConfig *config;

  self = GST_TENSORDEC_CAST (trans);

  g_assert (self->configured);

  config = &self->tensor_config;

  if (self->add_padding) {
    gsize offset;

    /** flag add_padding only for video */
    g_assert (self->output_type == OUTPUT_VIDEO);

    offset = config->info.dimension[0] * config->info.dimension[1];

    if (offset % 4)
      offset += 4 - (offset % 4);

    *othersize = offset * config->info.dimension[2] * config->info.dimension[3];
  } else {
    *othersize = size;
  }

  return TRUE;
}

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
gst_tensordec_plugin_init (GstPlugin * plugin)
{
  /**
   * debug category for fltering log messages
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensordec_debug, "tensor_decoder",
      0, "Element to convert tensor to media stream");

  return gst_element_register (plugin, "tensor_decoder", GST_RANK_NONE,
      GST_TYPE_TENSORDEC);
}

/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensor_decoder"
#endif

/**
 * gstreamer looks for this structure to register tensor_decoder
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_decoder,
    "Element to convert tensor to media stream",
    gst_tensordec_plugin_init,
    VERSION, "LGPL", "GStreamer", "http://gstreamer.net/");
