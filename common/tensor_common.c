/**
 * NNStreamer Common Header's Contents
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
 * @file	tensor_common.c
 * @date	29 May 2018
 * @brief	Common data for NNStreamer, the GStreamer plugin for neural networks
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <tensor_common.h>
#include <string.h>
#include <glib.h>

/**
 * @brief String representations for each tensor element type.
 */
const gchar *tensor_element_typename[] = {
  [_NNS_INT32] = "int32",
  [_NNS_UINT32] = "uint32",
  [_NNS_INT16] = "int16",
  [_NNS_UINT16] = "uint16",
  [_NNS_INT8] = "int8",
  [_NNS_UINT8] = "uint8",
  [_NNS_FLOAT64] = "float64",
  [_NNS_FLOAT32] = "float32",
  [_NNS_INT64] = "int64",
  [_NNS_UINT64] = "uint64",
  [_NNS_END] = NULL,
};

/**
 * @brief Get media type from structure
 * @param structure structure to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_END for unsupported type)
 */
media_type
gst_tensor_media_type_from_structure (const GstStructure * structure)
{
  const gchar *name;

  name = gst_structure_get_name (structure);

  g_return_val_if_fail (name != NULL, _NNS_MEDIA_END);

  /** @todo Support other types */
  if (g_str_has_prefix (name, "video/")) {
    return _NNS_VIDEO;
  } else if (g_str_has_prefix (name, "audio/")) {
    return _NNS_AUDIO;
  }

  /** unknown, or not-supported type */
  return _NNS_MEDIA_END;
}

/**
 * @brief Get media type from caps
 * @param caps caps to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_END for unsupported type)
 */
media_type
gst_tensor_media_type_from_caps (const GstCaps * caps)
{
  GstStructure *structure;

  structure = gst_caps_get_structure (caps, 0);

  return gst_tensor_media_type_from_structure (structure);
}

/**
 * @brief Initialize the video info structure
 * @param v_info video info structure to be initialized
 */
void
gst_tensor_video_info_init (GstTensorVideoInfo * v_info)
{
  g_return_if_fail (v_info != NULL);

  v_info->format = GST_VIDEO_FORMAT_UNKNOWN;
  v_info->w = 0;
  v_info->h = 0;
  v_info->fn = -1;
  v_info->fd = -1;
}

/**
 * @brief Initialize the audio info structure
 * @param a_info audio info structure to be initialized
 */
void
gst_tensor_audio_info_init (GstTensorAudioInfo * a_info)
{
  g_return_if_fail (a_info != NULL);

  a_info->format = GST_AUDIO_FORMAT_UNKNOWN;
  a_info->ch = 0;
  a_info->rate = 0;
  a_info->frames = 0;
}

/**
 * @brief Set video info to configure tensor
 * @param v_info video info structure to be filled
 * @param structure caps structure
 */
void
gst_tensor_video_info_from_structure (GstTensorVideoInfo * v_info,
    const GstStructure * structure)
{
  const gchar *format;

  g_return_if_fail (v_info != NULL);
  g_return_if_fail (structure != NULL);

  gst_tensor_video_info_init (v_info);

  format = gst_structure_get_string (structure, "format");
  if (format) {
    v_info->format = gst_video_format_from_string (format);
  }

  gst_structure_get_int (structure, "width", &v_info->w);
  gst_structure_get_int (structure, "height", &v_info->h);
  gst_structure_get_fraction (structure, "framerate", &v_info->fn, &v_info->fd);
}

/**
 * @brief Set audio info to configure tensor
 * @param a_info audio info structure to be filled
 * @param structure caps structure
 */
void
gst_tensor_audio_info_from_structure (GstTensorAudioInfo * a_info,
    const GstStructure * structure)
{
  const gchar *format;

  g_return_if_fail (a_info != NULL);
  g_return_if_fail (structure != NULL);

  gst_tensor_audio_info_init (a_info);

  format = gst_structure_get_string (structure, "format");
  if (format) {
    a_info->format = gst_audio_format_from_string (format);
  }

  gst_structure_get_int (structure, "channels", &a_info->ch);
  gst_structure_get_int (structure, "rate", &a_info->rate);

  /**
   * @todo How can we get the frames per buffer?
   * It depends on the tensorflow model, so simply set sample rate.
   */
  a_info->frames = a_info->rate;
}

/**
 * @brief Set the video info structure from tensor config
 * @param v_info video info structure to be filled
 * @param config tensor config structure to be interpreted
 * @return TRUE if supported format
 */
gboolean
gst_tensor_video_info_from_config (GstTensorVideoInfo * v_info,
    const GstTensorConfig * config)
{
  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (v_info != NULL, FALSE);

  gst_tensor_video_info_init (v_info);

  g_return_val_if_fail (config->tensor_media_type == _NNS_VIDEO, FALSE);

  v_info->format = (GstVideoFormat) config->tensor_media_format;
  v_info->w = config->dimension[1];
  v_info->h = config->dimension[2];
  v_info->fn = config->rate_n;
  v_info->fd = config->rate_d;

  return (v_info->format != GST_VIDEO_FORMAT_UNKNOWN);
}

/**
 * @brief Set the audio info structure from tensor config
 * @param a_info audio info structure to be filled
 * @param config tensor config structure to be interpreted
 * @return TRUE if supported format
 */
gboolean
gst_tensor_audio_info_from_config (GstTensorAudioInfo * a_info,
    const GstTensorConfig * config)
{
  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (a_info != NULL, FALSE);

  gst_tensor_audio_info_init (a_info);

  g_return_val_if_fail (config->tensor_media_type == _NNS_AUDIO, FALSE);

  a_info->format = (GstAudioFormat) config->tensor_media_format;
  a_info->ch = config->dimension[0];
  a_info->frames = config->dimension[1];
  a_info->rate = config->rate_n;

  return (a_info->format != GST_AUDIO_FORMAT_UNKNOWN);
}

/**
 * @brief Initialize the tensor config info structure
 * @param config tensor config structure to be initialized
 */
void
gst_tensor_config_init (GstTensorConfig * config)
{
  guint i;

  g_return_if_fail (config != NULL);

  config->rank = 0;
  config->type = _NNS_END;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->dimension[i] = 0;
  }

  config->rate_n = -1;
  config->rate_d = -1;
  config->frame_size = 0;
  config->tensor_media_type = _NNS_MEDIA_END;
  config->tensor_media_format = 0;
}

/**
 * @brief Check the tensor is all configured
 * @param config tensor config structure
 * @return TRUE if configured
 */
gboolean
gst_tensor_config_validate (GstTensorConfig * config)
{
  guint i;

  g_return_val_if_fail (config != NULL, FALSE);

  if (config->rank == 0 || config->type == _NNS_END) {
    return FALSE;
  }

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (config->dimension[i] == 0) {
      return FALSE;
    }
  }

  if (config->rate_n < 0 || config->rate_d < 0) {
    return FALSE;
  }

  if (config->frame_size == 0 || config->tensor_media_format == 0) {
    return FALSE;
  }

  switch (config->tensor_media_type) {
    case _NNS_VIDEO:
    case _NNS_AUDIO:
      break;
    default:
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Compare tensor config info
 * @param TRUE if same
 */
gboolean
gst_tensor_config_is_same (const GstTensorConfig * c1,
    const GstTensorConfig * c2)
{
  guint i;

  g_return_val_if_fail (c1 != NULL, FALSE);
  g_return_val_if_fail (c2 != NULL, FALSE);

  if (c1->tensor_media_type != c2->tensor_media_type) {
    return FALSE;
  }

  if (c1->tensor_media_format != c2->tensor_media_format) {
    return FALSE;
  }

  if (c1->rank == c2->rank && c1->type == c2->type &&
      c1->rate_n == c2->rate_n && c1->rate_d == c2->rate_d &&
      c1->frame_size == c2->frame_size) {
    for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
      if (c1->dimension[i] != c2->dimension[i]) {
        return FALSE;
      }
    }

    /** matched all */
    return TRUE;
  }

  return FALSE;
}

/**
 * @brief Parse structure and set tensor config info
 * @param config tensor config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if ok
 */
static gboolean
gst_tensor_config_from_tensor_structure (GstTensorConfig * config,
    const GstStructure * structure)
{
  const gchar *type_string;

  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (structure != NULL, FALSE);

  gst_tensor_config_init (config);

  if (!gst_structure_has_name (structure, "other/tensor")) {
    err_print ("caps is not tensor %s\n", gst_structure_get_name (structure));
    return FALSE;
  }

  gst_structure_get_int (structure, "rank", &config->rank);
  gst_structure_get_int (structure, "dim1", (gint *) (&config->dimension[0]));
  gst_structure_get_int (structure, "dim2", (gint *) (&config->dimension[1]));
  gst_structure_get_int (structure, "dim3", (gint *) (&config->dimension[2]));
  gst_structure_get_int (structure, "dim4", (gint *) (&config->dimension[3]));

  type_string = gst_structure_get_string (structure, "type");
  config->type = get_tensor_type (type_string);

  gst_structure_get_fraction (structure, "framerate", &config->rate_n,
      &config->rate_d);

  if (config->type != _NNS_END) {
    config->frame_size =
        tensor_element_size[config->type] *
        get_tensor_element_count (config->dimension);
  }

  /** @todo we cannot get media type from caps */
  if (config->rank == 2) {
    config->tensor_media_type = _NNS_AUDIO;
  } else if (config->rank == 3) {
    config->tensor_media_type = _NNS_VIDEO;
  }

  /** @todo we cannot get media format from caps */
  if (config->tensor_media_type == _NNS_VIDEO) {
    if (config->type == _NNS_UINT8) {
      switch (config->dimension[0]) {
        case 3:
          config->tensor_media_format = GST_VIDEO_FORMAT_RGB;
          break;
        case 4:
          config->tensor_media_format = GST_VIDEO_FORMAT_BGRx;
          break;
        default:
          config->tensor_media_format = GST_VIDEO_FORMAT_UNKNOWN;
          break;
      }
    }
  } else if (config->tensor_media_type == _NNS_AUDIO) {
    switch (config->type) {
      case _NNS_INT8:
        config->tensor_media_format = GST_AUDIO_FORMAT_S8;
        break;
      case _NNS_UINT8:
        config->tensor_media_format = GST_AUDIO_FORMAT_U8;
        break;
      case _NNS_INT16:
        config->tensor_media_format = GST_AUDIO_FORMAT_S16;
        break;
      case _NNS_UINT16:
        config->tensor_media_format = GST_AUDIO_FORMAT_U16;
        break;
      default:
        config->tensor_media_format = GST_AUDIO_FORMAT_UNKNOWN;
        break;
    }
  }

  return TRUE;
}

/**
 * @brief Parse caps and set tensor config info
 * @param config tensor config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if ok
 */
static gboolean
gst_tensor_config_from_media_structure (GstTensorConfig * config,
    const GstStructure * structure)
{
  media_type m_type;

  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (structure != NULL, FALSE);

  gst_tensor_config_init (config);

  m_type = gst_tensor_media_type_from_structure (structure);

  if (m_type == _NNS_VIDEO) {
    GstTensorVideoInfo v_info;

    gst_tensor_video_info_from_structure (&v_info, structure);
    gst_tensor_config_from_video_info (config, &v_info);
  } else if (m_type == _NNS_AUDIO) {
    GstTensorAudioInfo a_info;

    gst_tensor_audio_info_from_structure (&a_info, structure);
    gst_tensor_config_from_audio_info (config, &a_info);
  } else {
    /** @todo Support other types */
    err_print ("Unsupported type %d\n", m_type);
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Parse structure and set tensor config info
 * @param config tensor config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if ok
 */
gboolean
gst_tensor_config_from_structure (GstTensorConfig * config,
    const GstStructure * structure)
{
  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (structure != NULL, FALSE);

  if (gst_structure_has_name (structure, "other/tensor")) {
    return gst_tensor_config_from_tensor_structure (config, structure);
  } else {
    return gst_tensor_config_from_media_structure (config, structure);
  }
}

/**
 * @brief Set the tensor config structure from video info
 * @param config tensor config structure to be filled
 * @param v_info video info structure to be interpreted
 * @return TRUE if ok
 */
gboolean
gst_tensor_config_from_video_info (GstTensorConfig * config,
    const GstTensorVideoInfo * v_info)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/image
   * A 4-D uint8 or float32 Tensor of shape [batch_size, height, width, channels]
   * where channels is 1, 3, or 4.
   */
  gboolean res = TRUE;

  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (v_info != NULL, FALSE);

  gst_tensor_config_init (config);

  /** [color-space][width][height] */
  config->rank = 3;

  switch (v_info->format) {
    case GST_VIDEO_FORMAT_RGB:
      config->type = _NNS_UINT8;
      config->dimension[0] = 3;
      break;
    case GST_VIDEO_FORMAT_BGRx:
      config->type = _NNS_UINT8;
      config->dimension[0] = 4;
      break;
    default:
      /** unsupported format */
      err_print ("Unsupported format = %d\n", v_info->format);
      res = FALSE;
      break;
  }

  config->dimension[1] = v_info->w;
  config->dimension[2] = v_info->h;
  config->dimension[3] = 1;

  config->rate_n = v_info->fn;
  config->rate_d = v_info->fd;

  if (config->type != _NNS_END) {
    config->frame_size =
        tensor_element_size[config->type] *
        get_tensor_element_count (config->dimension);
  }

  config->tensor_media_type = _NNS_VIDEO;
  config->tensor_media_format = v_info->format;
  return res;
}

/**
 * @brief Set the tensor config structure from audio info
 * @param config tensor config structure to be filled
 * @param a_info audio info structure to be interpreted
 * @return TRUE if ok
 */
gboolean
gst_tensor_config_from_audio_info (GstTensorConfig * config,
    const GstTensorAudioInfo * a_info)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/audio
   * A 3-D float32 Tensor of shape [batch_size, frames, channels]
   * or a 2-D float32 Tensor of shape [batch_size, frames].
   */
  gboolean res = TRUE;

  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (a_info != NULL, FALSE);

  gst_tensor_config_init (config);

  /** [channels][frames] */
  config->rank = 2;

  switch (a_info->format) {
    case GST_AUDIO_FORMAT_S8:
      config->type = _NNS_INT8;
      break;
    case GST_AUDIO_FORMAT_U8:
      config->type = _NNS_UINT8;
      break;
    case GST_AUDIO_FORMAT_S16:
      config->type = _NNS_INT16;
      break;
    case GST_AUDIO_FORMAT_U16:
      config->type = _NNS_UINT16;
      break;
    default:
      /** unsupported format */
      err_print ("Unsupported format = %d\n", a_info->format);
      res = FALSE;
      break;
  }

  config->dimension[0] = a_info->ch;
  config->dimension[1] = a_info->frames;
  config->dimension[2] = 1;
  config->dimension[3] = 1;

  if (a_info->rate > 0) {
    config->rate_n = a_info->rate;
    config->rate_d = 1;
  }

  if (config->type != _NNS_END) {
    config->frame_size =
        tensor_element_size[config->type] *
        get_tensor_element_count (config->dimension);
  }

  config->tensor_media_type = _NNS_AUDIO;
  config->tensor_media_format = a_info->format;
  return res;
}

/**
 * @brief Get tensor caps from tensor config
 * @param config tensor config info
 * @return caps for given config
 */
GstCaps *
gst_tensor_caps_from_config (const GstTensorConfig * config)
{
  GstCaps *caps;
  GstStaticCaps raw_caps = GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT);

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_static_caps_get (&raw_caps);
  caps = gst_caps_make_writable (caps);

  if (config->rank > 0) {
    gst_caps_set_simple (caps, "rank", G_TYPE_INT, config->rank, NULL);
  }

  if (config->dimension[0] > 0) {
    gst_caps_set_simple (caps, "dim1", G_TYPE_INT, config->dimension[0], NULL);
  }

  if (config->dimension[1] > 0) {
    gst_caps_set_simple (caps, "dim2", G_TYPE_INT, config->dimension[1], NULL);
  }

  if (config->dimension[2] > 0) {
    gst_caps_set_simple (caps, "dim3", G_TYPE_INT, config->dimension[2], NULL);
  }

  if (config->dimension[3] > 0) {
    gst_caps_set_simple (caps, "dim4", G_TYPE_INT, config->dimension[3], NULL);
  }

  if (config->type != _NNS_END) {
    gst_caps_set_simple (caps, "type", G_TYPE_STRING,
        tensor_element_typename[config->type], NULL);
  }

  if (config->rate_n > 0 && config->rate_d > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
  }

  return gst_caps_simplify (caps);
}

/**
 * @brief Get video caps from tensor config
 * @param config tensor config info
 * @return caps for given config
 */
static GstCaps *
gst_tensor_video_caps_from_config (const GstTensorConfig * config)
{
  GstTensorVideoInfo v_info;
  GstCaps *caps;
  GstStaticCaps raw_caps = GST_STATIC_CAPS (GST_TENSOR_VIDEO_CAPS_STR);

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_static_caps_get (&raw_caps);
  caps = gst_caps_make_writable (caps);

  gst_tensor_video_info_from_config (&v_info, config);

  if (v_info.format != GST_VIDEO_FORMAT_UNKNOWN) {
    const gchar *format_string = gst_video_format_to_string (v_info.format);
    gst_caps_set_simple (caps, "format", G_TYPE_STRING, format_string, NULL);
  }

  if (v_info.w > 0) {
    gst_caps_set_simple (caps, "width", G_TYPE_INT, v_info.w, NULL);
  }

  if (v_info.h > 0) {
    gst_caps_set_simple (caps, "height", G_TYPE_INT, v_info.h, NULL);
  }

  if (v_info.fn > 0 && v_info.fd > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION,
        v_info.fn, v_info.fd, NULL);
  }

  return gst_caps_simplify (caps);
}

/**
 * @brief Get audio caps from tensor config
 * @param config tensor config info
 * @return caps for given config
 */
static GstCaps *
gst_tensor_audio_caps_from_config (const GstTensorConfig * config)
{
  GstTensorAudioInfo a_info;
  GstCaps *caps;
  GstStaticCaps raw_caps = GST_STATIC_CAPS (GST_TENSOR_AUDIO_CAPS_STR);

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_static_caps_get (&raw_caps);
  caps = gst_caps_make_writable (caps);

  gst_tensor_audio_info_from_config (&a_info, config);

  if (a_info.format != GST_AUDIO_FORMAT_UNKNOWN) {
    const gchar *format_string = gst_audio_format_to_string (a_info.format);
    gst_caps_set_simple (caps, "format", G_TYPE_STRING, format_string, NULL);
  }

  if (a_info.ch > 0) {
    gst_caps_set_simple (caps, "channels", G_TYPE_INT, a_info.ch, NULL);
  }

  if (a_info.rate > 0) {
    gst_caps_set_simple (caps, "rate", G_TYPE_INT, a_info.rate, NULL);
  }

  return gst_caps_simplify (caps);
}

/**
 * @brief Get media caps from tensor config
 * @param config tensor config info
 * @return caps for given config
 */
GstCaps *
gst_tensor_media_caps_from_config (const GstTensorConfig * config)
{
  GstCaps *caps = NULL;

  g_return_val_if_fail (config != NULL, NULL);

  switch (config->tensor_media_type) {
    case _NNS_VIDEO:
      caps = gst_tensor_video_caps_from_config (config);
      break;
    case _NNS_AUDIO:
      caps = gst_tensor_audio_caps_from_config (config);
      break;
    default:
      /** @todo Support other types */
      err_print ("Unsupported type %d\n", config->tensor_media_type);
      break;
  }

  return caps;
}

/**
 * @brief Determine if we need zero-padding
 * @return 1 if we need to add (or remove) stride per row from the stream data. 0 otherwise.
 */
gint
gst_tensor_video_stride_padding_per_row (GstVideoFormat format, gint width)
{
  /** @todo The actual list is much longer. fill them (read https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html ) */
  switch (format) {
    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
    case GST_VIDEO_FORMAT_I420:
      if (width % 4) {
        return 1;
      }
      break;
    default:
      break;
  }

  return 0;
}

/**
 * @brief Get tensor_type from string tensor_type input
 * @return Corresponding tensor_type. _NNS_END if unrecognized value is there.
 * @param typestr The string type name, supposed to be one of tensor_element_typename[]
 */
tensor_type
get_tensor_type (const gchar * typestr)
{
  int len;

  if (!typestr)
    return _NNS_END;
  len = strlen (typestr);

  if (typestr[0] == 'u' || typestr[0] == 'U') {
    /**
     * Let's believe the developer and the following three letters are "int"
     * (case insensitive)
     */
    if (len == 6) {             /* uint16, uint32 */
      if (typestr[4] == '1' && typestr[5] == '6')
        return _NNS_UINT16;
      else if (typestr[4] == '3' && typestr[5] == '2')
        return _NNS_UINT32;
      else if (typestr[4] == '6' && typestr[5] == '4')
        return _NNS_UINT64;
    } else if (len == 5) {      /* uint8 */
      if (typestr[4] == '8')
        return _NNS_UINT8;
    }
  } else if (typestr[0] == 'i' || typestr[0] == 'I') {
    /**
     * Let's believe the developer and the following two letters are "nt"
     * (case insensitive)
     */
    if (len == 5) {             /* int16, int32 */
      if (typestr[3] == '1' && typestr[4] == '6')
        return _NNS_INT16;
      else if (typestr[3] == '3' && typestr[4] == '2')
        return _NNS_INT32;
      else if (typestr[3] == '6' && typestr[4] == '4')
        return _NNS_INT64;
    } else if (len == 4) {      /* int8 */
      if (typestr[3] == '8')
        return _NNS_INT8;
    }
    return _NNS_END;
  } else if (typestr[0] == 'f' || typestr[0] == 'F') {
    /* Let's assume that the following 4 letters are "loat" */
    if (len == 7) {
      if (typestr[5] == '6' && typestr[6] == '4')
        return _NNS_FLOAT64;
      else if (typestr[5] == '3' && typestr[6] == '2')
        return _NNS_FLOAT32;
    }
  }

  return _NNS_END;
}

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index. Returns -1 if not found.
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
int
find_key_strv (const gchar ** strv, const gchar * key)
{
  int cursor = 0;

  g_assert (strv != NULL);
  while (strv[cursor]) {
    if (!g_ascii_strcasecmp (strv[cursor], key))
      return cursor;
    cursor++;
  }

  return -1;                    /* Not Found */
}

/**
 * @brief Parse tensor dimension parameter string
 * @return The Rank. 0 if error.
 * @param param The parameter string in the format of d1:d2:d3:d4, d1:d2:d3, d1:d2, or d1, where dN is a positive integer and d1 is the innermost dimension; i.e., dim[d4][d3][d2][d1];
 */
int
get_tensor_dimension (const gchar * param, uint32_t dim[NNS_TENSOR_RANK_LIMIT])
{
  gchar **strv = g_strsplit (param, ":", NNS_TENSOR_RANK_LIMIT);
  int i, retval = 0;
  guint64 val;

  g_assert (strv != NULL);

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (strv[i] == NULL)
      break;
    val = g_ascii_strtoull (strv[i], NULL, 10);
    dim[i] = val;
    retval = i + 1;
  }
  for (; i < NNS_TENSOR_RANK_LIMIT; i++)
    dim[i] = 1;

  g_strfreev (strv);
  return retval;
}

/**
 * @brief Count the number of elemnts of a tensor
 * @return The number of elements. 0 if error.
 * @param dim The tensor dimension
 */
size_t
get_tensor_element_count (const uint32_t dim[NNS_TENSOR_RANK_LIMIT])
{
  size_t count = 1;
  int i;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    count *= dim[i];
  }

  return count;
}

/**
 * @brief Extract other/tensor dim/type from GstStructure
 */
GstTensor_Filter_CheckStatus
get_tensor_from_structure (const GstStructure * str, tensor_dim dim,
    tensor_type * type, int *framerate_num, int *framerate_denum)
{
  GstTensor_Filter_CheckStatus ret = _TFC_INIT;
  const gchar *strval;
  int rank;
  int j;
  gint fn, fd;

  if (!gst_structure_has_name (str, "other/tensor"))
    return ret;

  if (gst_structure_get_int (str, "dim1", (int *) &dim[0]) &&
      gst_structure_get_int (str, "dim2", (int *) &dim[1]) &&
      gst_structure_get_int (str, "dim3", (int *) &dim[2]) &&
      gst_structure_get_int (str, "dim4", (int *) &dim[3])) {
    ret |= _TFC_DIMENSION;
    if (gst_structure_get_int (str, "rank", &rank)) {
      for (j = rank; j < NNS_TENSOR_RANK_LIMIT; j++)
        g_assert (dim[j] == 1);
    }
  }
  strval = gst_structure_get_string (str, "type");
  if (strval) {
    *type = get_tensor_type (strval);
    g_assert (*type != _NNS_END);
    ret |= _TFC_TYPE;
  }
  if (gst_structure_get_fraction (str, "framerate", &fn, &fd)) {
    if (framerate_num)
      *framerate_num = fn;
    if (framerate_denum)
      *framerate_denum = fd;
    ret |= _TFC_FRAMERATE;
  }
  return ret;
}

/**
 * @brief internal static function to trim the front.
 */
static const gchar *
ftrim (const gchar * str)
{
  if (!str)
    return str;
  while (*str && (*str == ' ' || *str == '\t')) {
    str++;
  }
  return str;
}

/**
 * @brief Extract other/tensors dim/type from GstStructure
 */
int
get_tensors_from_structure (const GstStructure * str,
    GstTensor_TensorsMeta * meta, int *framerate_num, int *framerate_denom)
{
  int num = 0;
  int rank = 0;
  const gchar *strval;
  gint fn = 0, fd = 0;
  gchar **strv;
  int counter = 0;

  if (!gst_structure_has_name (str, "other/tensors"))
    return 0;

  if (gst_structure_get_int (str, "num_tensors", (int *) &num)) {
    if (num > 16 || num < 1)
      num = 0;
  }
  if (0 == num)
    return 0;

  meta->num_tensors = num;

  if (gst_structure_get_int (str, "rank", (int *) &rank)) {
    if (rank != NNS_TENSOR_RANK_LIMIT) {
      err_print ("rank value of other/tensors incorrect.\n");
      rank = 0;
    }
  }
  if (0 == rank)
    return 0;

  if (gst_structure_get_fraction (str, "framerate", &fn, &fd)) {
    if (framerate_num)
      *framerate_num = fn;
    if (framerate_denom)
      *framerate_denom = fd;
  }

  strval = gst_structure_get_string (str, "dimensions");
  strv = g_strsplit (strval, ",", -1);
  counter = 0;
  while (strv[counter]) {
    int ret;

    if (counter >= num) {
      err_print
          ("The number of dimensions does not match the number of tensors.\n");
      return 0;
    }
    ret = get_tensor_dimension (ftrim (strv[counter]), meta->dims[counter]);
    if (ret > NNS_TENSOR_RANK_LIMIT || ret < 1)
      return 0;
    counter++;
  }
  if (counter != num) {
    err_print
        ("The number of dimensions does not match the number of tensors.\n");
    return 0;
  }
  g_strfreev (strv);

  strval = gst_structure_get_string (str, "types");
  strv = g_strsplit (strval, ",", -1);
  counter = 0;
  while (strv[counter]) {
    if (counter >= num) {
      err_print ("The number of types does not match the number of tensors.\n");
      return 0;
    }
    meta->types[counter] = get_tensor_type (ftrim (strv[counter]));
    if (meta->types[counter] >= _NNS_END)
      return 0;
    counter++;
  }
  if (counter != num) {
    err_print ("The number of types does not match the number of tensors.\n");
    return 0;
  }
  g_strfreev (strv);
  return num;
}

/**
 * @brief Get tensor dimension/type from GstCaps
 */
GstTensor_Filter_CheckStatus
get_tensor_from_padcap (const GstCaps * caps, tensor_dim dim,
    tensor_type * type, int *framerate_num, int *framerate_denum)
{
  GstTensor_Filter_CheckStatus ret = _TFC_INIT;
  unsigned int i, capsize;
  const GstStructure *str;
  gint fn = 0, fd = 0;

  g_assert (NNS_TENSOR_RANK_LIMIT == 4);        /* This code assumes rank limit is 4 */
  if (!caps)
    return ret;

  capsize = gst_caps_get_size (caps);
  for (i = 0; i < capsize; i++) {
    str = gst_caps_get_structure (caps, i);

    tensor_dim _dim;
    tensor_type _type = _NNS_END;
    int _fn, _fd;

    GstTensor_Filter_CheckStatus tmpret = get_tensor_from_structure (str,
        _dim, &_type, &_fn, &_fd);

    /**
     * Already cofnigured and more cap info is coming.
     * I'm not sure how this happens, but let's be ready for this.
     */
    if (tmpret & _TFC_DIMENSION) {
      if (ret & _TFC_DIMENSION) {
        g_assert (0 == memcmp (_dim, dim, sizeof (tensor_dim)));
      } else {
        memcpy (dim, _dim, sizeof (tensor_dim));
        ret |= _TFC_DIMENSION;
      }
    }

    if (tmpret & _TFC_TYPE) {
      if (ret & _TFC_TYPE) {
        g_assert (*type == _type);
      } else {
        *type = _type;
        ret |= _TFC_TYPE;
      }
    }

    if (tmpret & _TFC_FRAMERATE) {
      if (ret & _TFC_FRAMERATE) {
        g_assert (fn == _fn && fd == _fd);
      } else {
        fn = _fn;
        fd = _fd;
        if (framerate_num)
          *framerate_num = fn;
        if (framerate_denum)
          *framerate_denum = fd;
        ret |= _TFC_FRAMERATE;
      }
    }
  }
  return ret;
}

/**
 * @brief A callback for typefind, trying to find whether a file is other/tensors or not.
 * For the concrete definition of headers, please look at the wiki page of nnstreamer:
 * https://github.com/nnsuite/nnstreamer/wiki/Design-External-Save-Format-for-other-tensor-and-other-tensors-Stream-for-TypeFind
 */
void
gst_tensors_typefind_function (GstTypeFind * tf, gpointer pdata)
{
  const guint8 *data = gst_type_find_peek (tf, 0, 40);  /* The first 40 bytes are header-0 in v.1 protocol */
  const char formatstr[] = "TENSORST";
  const unsigned int *supported_version = (const unsigned int *) (&data[8]);
  const unsigned int *num_tensors = (const unsigned int *) (&data[12]);
  if (data &&
      memcmp (data, formatstr, 8) == 0 &&
      *supported_version == 1 && *num_tensors <= 16 && *num_tensors >= 1) {
    gst_type_find_suggest (tf, GST_TYPE_FIND_MAXIMUM,
        gst_caps_new_simple ("other/tensorsave", NULL, NULL));
  }
}
