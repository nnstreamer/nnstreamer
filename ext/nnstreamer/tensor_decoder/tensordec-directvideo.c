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
 * @bug		No known bugs except for NYI items
 *
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

#define DECODER_DV_VIDEO_CAPS_STR \
    GST_VIDEO_CAPS_MAKE ("{ GRAY8, RGB, BGR, RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR }") \
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
      case DIRECT_VIDEO_FORMAT_UNKNOWN:
        GST_WARNING ("Default format has been applied: GRAY8");
        format = GST_VIDEO_FORMAT_GRAY8;
        break;
      default:
        GST_ERROR ("Invalid format. Please check the video format");
        return NULL;
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
        GST_WARNING ("Default format has been applied: RGB");
        format = GST_VIDEO_FORMAT_RGB;
        break;
      default:
        GST_ERROR ("Invalid format. Please check the video format");
        return NULL;
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
        GST_WARNING ("Default format has been applied: BGRx");
        format = GST_VIDEO_FORMAT_BGRx;
        break;
      default:
        GST_ERROR ("Invalid format. Please check the video format");
        return NULL;
    }
  } else {
    GST_ERROR ("%d channel is not supported", channel);
    return NULL;
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
_get_video_xraw_bufsize (const tensor_dim dim)
{
  /* dim[0] is bpp and there is zeropadding only when dim[0]%4 > 0 */
  return (size_t)((dim[0] * dim[1] - 1) / 4 + 1) * 4 * dim[2];
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
dv_getTransformSize (void **pdata, const GstTensorsConfig * config,
    GstCaps * caps, size_t size, GstCaps * othercaps, GstPadDirection direction)
{
  /* Direct video uses the first tensor only even if it's multi-tensor */
  const uint32_t *dim = &(config->info.info[0].dimension[0]);
  UNUSED (pdata);
  UNUSED (caps);
  UNUSED (size);
  UNUSED (othercaps);

  if (direction == GST_PAD_SINK)
    return _get_video_xraw_bufsize (dim);
  else
    return 0; /** @todo NYI */
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
  size_t size = _get_video_xraw_bufsize (dim);
  UNUSED (pdata);

  g_assert (outbuf);
  if (gst_buffer_get_size (outbuf) > 0 && gst_buffer_get_size (outbuf) != size) {
    gst_buffer_set_size (outbuf, size);
  }
  g_assert (config->info.info[0].type == _NNS_UINT8);

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
    gst_memory_unref (out_mem);

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
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_dv (void)
{
  nnstreamer_decoder_exit (directVideo.modename);
}
