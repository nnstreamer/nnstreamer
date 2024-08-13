/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "direct video"
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 *
 */
/**
 * @file	tensordec-directvideo.c
 * @date	04 Nov 2018
 * @brief	NNStreamer tensor-decoder subplugin, "direct video",
 *              which converts tensors to video directly.
 *
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		If the element size is 2 or larger, padding won't work.
 *              GRAY16 types has size of 2 and if you have padding, it won't work.
 *              To correct this, dv_decode() should be fixed.
 */

#include <string.h>
#include <glib.h>
#include <gst/video/video-format.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "tensordecutil.h"

void init_dv (void) __attribute__ ((constructor));
void fini_dv (void) __attribute__ ((destructor));

#define DECODER_DV_FORMATS "{ GRAY8, RGB, BGR, RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR, GRAY16_BE, GRAY16_LE }"

#define DECODER_DV_VIDEO_CAPS_STR \
    GST_VIDEO_CAPS_MAKE (DECODER_DV_FORMATS) \
    ", views = (int) 1, interlace-mode = (string) progressive"

/**
 * @brief The supported video formats
 */
typedef enum
{
  DIRECT_VIDEO_FORMAT_UNKNOWN = 0,

  /* Single Channel, Default: GRAY8 */
  DIRECT_VIDEO_FORMAT_GRAY8 = 1,

  /* 3 Channels, Default: RGB */
  DIRECT_VIDEO_FORMAT_RGB = 2,
  DIRECT_VIDEO_FORMAT_BGR = 3,

  /* 4 Channels, Default: BGRx */
  DIRECT_VIDEO_FORMAT_RGBx = 4,
  DIRECT_VIDEO_FORMAT_BGRx = 5,
  DIRECT_VIDEO_FORMAT_xRGB = 6,
  DIRECT_VIDEO_FORMAT_xBGR = 7,
  DIRECT_VIDEO_FORMAT_RGBA = 8,
  DIRECT_VIDEO_FORMAT_BGRA = 9,
  DIRECT_VIDEO_FORMAT_ARGB = 10,
  DIRECT_VIDEO_FORMAT_ABGR = 11,
  DIRECT_VIDEO_FORMAT_GRAY16_BE = 12,
  DIRECT_VIDEO_FORMAT_GRAY16_LE = 13,
} direct_video_formats;

/**
 * @brief Data structure for direct video options.
 */
typedef struct
{
  /* From option1 */
  direct_video_formats format;
} direct_video_ops;

/**
 * @brief List of the formats of direct video
 */
static const char *dv_formats[] = {
  [DIRECT_VIDEO_FORMAT_UNKNOWN] = "UNKNOWN",
  [DIRECT_VIDEO_FORMAT_GRAY8] = "GRAY8",
  [DIRECT_VIDEO_FORMAT_RGB] = "RGB",
  [DIRECT_VIDEO_FORMAT_BGR] = "BGR",
  [DIRECT_VIDEO_FORMAT_RGBx] = "RGBx",
  [DIRECT_VIDEO_FORMAT_BGRx] = "BGRx",
  [DIRECT_VIDEO_FORMAT_xRGB] = "xRGB",
  [DIRECT_VIDEO_FORMAT_xBGR] = "xBGR",
  [DIRECT_VIDEO_FORMAT_RGBA] = "RGBA",
  [DIRECT_VIDEO_FORMAT_BGRA] = "BGRA",
  [DIRECT_VIDEO_FORMAT_ARGB] = "ARGB",
  [DIRECT_VIDEO_FORMAT_ABGR] = "ABGR",
  [DIRECT_VIDEO_FORMAT_GRAY16_BE] = "GRAY16_BE",
  [DIRECT_VIDEO_FORMAT_GRAY16_LE] = "GRAY16_LE",
  NULL,
};

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
dv_init (void **pdata)
{
  direct_video_ops *ddata;
  ddata = *pdata = g_try_new0 (direct_video_ops, 1);

  if (ddata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  ddata->format = DIRECT_VIDEO_FORMAT_UNKNOWN;

  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
dv_exit (void **pdata)
{
  if (pdata)
    g_free (*pdata);
  return;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
dv_setOption (void **pdata, int opNum, const char *param)
{
  direct_video_ops *ddata;

  if (!pdata || !*pdata) {
    GST_ERROR ("There is no plugin data.");
    return FALSE;
  }

  if (NULL == param || *param == '\0') {
    GST_ERROR ("Please set the valid value at option.");
    return FALSE;
  }

  ddata = *pdata;

  /* When the dimension[0] is 4, the video format will be decided by option1. */
  switch (opNum) {
    case 0:
    {
      int f = find_key_strv (dv_formats, param);

      ddata->format = (f < 0) ? DIRECT_VIDEO_FORMAT_UNKNOWN : f;
      if (ddata->format == DIRECT_VIDEO_FORMAT_UNKNOWN) {
        return FALSE;
      }
      break;
    }
    default:
      break;
  }

  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
dv_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
  direct_video_ops *ddata = *pdata;
  /* Old gst_tensordec_video_caps_from_config () had this */
  GstVideoFormat format;
  gint width, height, channel;
  GstCaps *caps;
  guint element_size_from_cap = 1; /** Assume 1 byte per element */
  tensor_type input_tensor_type = config->info.info[0].type;

  g_return_val_if_fail (config != NULL, NULL);
  GST_INFO ("Num Tensors = %d", config->info.num_tensors);
  g_return_val_if_fail (config->info.num_tensors >= 1, NULL);

  /* Direct video uses the first tensor only even if it's multi-tensor */
  channel = config->info.info[0].dimension[0];
  if (channel == 1) {
    switch (ddata->format) {
      case DIRECT_VIDEO_FORMAT_GRAY8:
        format = GST_VIDEO_FORMAT_GRAY8;
        break;
      case DIRECT_VIDEO_FORMAT_GRAY16_BE:
        format = GST_VIDEO_FORMAT_GRAY16_BE;
        element_size_from_cap = 2;
        break;
      case DIRECT_VIDEO_FORMAT_GRAY16_LE:
        format = GST_VIDEO_FORMAT_GRAY16_LE;
        element_size_from_cap = 2;
        break;
      case DIRECT_VIDEO_FORMAT_UNKNOWN:
      default:
        GST_WARNING ("Default format has been applied: GRAY8");
        format = GST_VIDEO_FORMAT_GRAY8;
        break;
    }
  } else if (channel == 3) {
    switch (ddata->format) {
      case DIRECT_VIDEO_FORMAT_RGB:
        format = GST_VIDEO_FORMAT_RGB;
        break;
      case DIRECT_VIDEO_FORMAT_BGR:
        format = GST_VIDEO_FORMAT_BGR;
        break;
      case DIRECT_VIDEO_FORMAT_UNKNOWN:
      default:
        GST_WARNING ("Default format has been applied: RGB");
        format = GST_VIDEO_FORMAT_RGB;
    }
  } else if (channel == 4) {
    switch (ddata->format) {
      case DIRECT_VIDEO_FORMAT_RGBx:
        format = GST_VIDEO_FORMAT_RGBx;
        break;
      case DIRECT_VIDEO_FORMAT_BGRx:
        format = GST_VIDEO_FORMAT_BGRx;
        break;
      case DIRECT_VIDEO_FORMAT_xRGB:
        format = GST_VIDEO_FORMAT_xRGB;
        break;
      case DIRECT_VIDEO_FORMAT_xBGR:
        format = GST_VIDEO_FORMAT_xBGR;
        break;
      case DIRECT_VIDEO_FORMAT_RGBA:
        format = GST_VIDEO_FORMAT_RGBA;
        break;
      case DIRECT_VIDEO_FORMAT_BGRA:
        format = GST_VIDEO_FORMAT_BGRA;
        break;
      case DIRECT_VIDEO_FORMAT_ARGB:
        format = GST_VIDEO_FORMAT_ARGB;
        break;
      case DIRECT_VIDEO_FORMAT_ABGR:
        format = GST_VIDEO_FORMAT_ABGR;
        break;
      case DIRECT_VIDEO_FORMAT_UNKNOWN:
      default:
        GST_WARNING ("Default format has been applied: BGRx");
        format = GST_VIDEO_FORMAT_BGRx;
        break;
    }
  } else {
    GST_ERROR ("%d channel is not supported", channel);
    return NULL;
  }

  if (gst_tensor_get_element_size (input_tensor_type) != element_size_from_cap) {
    GST_ERROR ("The element size of input tensor (%" G_GSIZE_FORMAT
        " byte / %s) for tensor_decoder::direct_video must be same as the element size of output (%u byte / %s). Note that except for GrayScale-16 format, it should be 1 byte / element, normally; i.e., RGB, RGBA, and BGRx. It is recommended to convert to UINT8 tensor stream or UINT16 tensor stream (GreyScale-16) before tensor_decoder::direct_video.",
        gst_tensor_get_element_size (input_tensor_type),
        gst_tensor_get_type_string (input_tensor_type),
        element_size_from_cap,
        ((element_size_from_cap == 2) ? "uint16" : "uint8"));
    return NULL;
  }

  if (input_tensor_type != _NNS_UINT8 && input_tensor_type != _NNS_UINT16) {
    GST_WARNING
        ("The input tensor type for tensor_decoder::direct_video is recommended to be either UINT8 or UINT16. The current type, %s, does not incur buffer size mismatch, but the actual behavior might be inconsistent. Try UINT8/UINT16, which is meant to be the video/x-raw element type.",
        gst_tensor_get_type_string (input_tensor_type));
  }

  width = config->info.info[0].dimension[1];
  height = config->info.info[0].dimension[2];

  caps = gst_caps_from_string (DECODER_DV_VIDEO_CAPS_STR);

  if (format != GST_VIDEO_FORMAT_UNKNOWN) {
    const char *format_string = gst_video_format_to_string (format);
    gst_caps_set_simple (caps, "format", G_TYPE_STRING, format_string, NULL);
  }

  if (width > 0) {
    gst_caps_set_simple (caps, "width", G_TYPE_INT, width, NULL);
  }

  if (height > 0) {
    gst_caps_set_simple (caps, "height", G_TYPE_INT, height, NULL);
  }

  setFramerateFromConfig (caps, config);

  return gst_caps_simplify (caps);
}

/** @brief get video output buffer size */
static size_t
_get_video_xraw_bufsize (const tensor_dim dim, gsize data_size)
{
  /* dim[0] is bpp and there is zeropadding only when dim[0]%4 > 0 */
  return (size_t) ((dim[0] * dim[1] - 1) / 4 + 1) * 4 * dim[2] * data_size;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
dv_getTransformSize (void **pdata, const GstTensorsConfig * config,
    GstCaps * caps, size_t size, GstCaps * othercaps, GstPadDirection direction)
{
  /* Direct video uses the first tensor only even if it's multi-tensor */
  const uint32_t *dim = &(config->info.info[0].dimension[0]);
  gsize data_size = gst_tensor_get_element_size (config->info.info[0].type);
  gsize transform_size = 0;
  UNUSED (pdata);
  UNUSED (caps);
  UNUSED (size);
  UNUSED (othercaps);

  if (direction == GST_PAD_SINK)
    transform_size = _get_video_xraw_bufsize (dim, data_size);

  return transform_size;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
dv_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  GstMapInfo out_info;
  GstMemory *out_mem;
  /* Direct video uses the first tensor only even if it's multi-tensor */
  const uint32_t *dim = &(config->info.info[0].dimension[0]);
  gsize data_size = gst_tensor_get_element_size (config->info.info[0].type);

  size_t size = _get_video_xraw_bufsize (dim, data_size);
  UNUSED (pdata);

  g_assert (outbuf);
  if (gst_buffer_get_size (outbuf) > 0 && gst_buffer_get_size (outbuf) != size) {
    gst_buffer_set_size (outbuf, size);
  }

  if (gst_buffer_get_size (outbuf) == 0) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    /* Don't reallocate. Reuse what's already given */
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    gst_memory_unref (out_mem);
    ml_loge ("Cannot map output memory / tensordec-directvideo.\n");
    return GST_FLOW_ERROR;
  }

  if (0 == ((dim[0] * dim[1]) % 4)) {
    /* No Padding Required */
    memcpy (out_info.data, input->data, input->size);
  } else {
    /* Do Padding */
    unsigned int h;
    uint8_t *ptr, *inp;

    ptr = (uint8_t *) out_info.data;
    inp = (uint8_t *) input->data;
    for (h = 0; h < dim[2]; h++) {
      memcpy (ptr, inp, (size_t) dim[0] * dim[1]);
      inp += (dim[0] * dim[1]);
      ptr += ((dim[0] * dim[1] - 1) / 4 + 1) * 4;
    }
  }
  gst_memory_unmap (out_mem, &out_info);

  if (gst_buffer_get_size (outbuf) == 0)
    gst_buffer_append_memory (outbuf, out_mem);
  else
    gst_buffer_replace_all_memory (outbuf, out_mem);

  /** @todo Caller of dv_decode in tensordec.c should call gst_memory_unmap to inbuf */

  return GST_FLOW_OK;
}

static gchar decoder_subplugin_direct_video[] = "direct_video";

/** @brief Direct-Video tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef directVideo = {
  .modename = decoder_subplugin_direct_video,
  .init = dv_init,
  .exit = dv_exit,
  .setOption = dv_setOption,
  .getOutCaps = dv_getOutCaps,
  .getTransformSize = dv_getTransformSize,
  .decode = dv_decode
};

/** @brief Initialize this object for tensordec-plugin */
void
init_dv (void)
{
  nnstreamer_decoder_probe (&directVideo);
  nnstreamer_decoder_set_custom_property_desc (decoder_subplugin_direct_video,
      "option1",
      "The output video format. If this is unspecified, it is 'GRAY8' (dim[0]/channel == 1), 'RGB' (dim[0]/channel == 3), or 'BGRx' (dim[0]/channel == 4). Available options are: "
      DECODER_DV_FORMATS, NULL);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_dv (void)
{
  nnstreamer_decoder_exit (directVideo.modename);
}
