/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "direct video"
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 * @file	tensordec-directvideo.c
 * @date	04 Nov 2018
 * @brief	NNStreamer tensor-decoder subplugin, "direct video",
 *              which converts tensors to video directly.
 *
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <string.h>
#include <glib.h>
#include "tensordec.h"
#include <tensor_common.h>

/** @brief tensordec-plugin's TensorDecDef callback */
static gboolean
dv_init (GstTensorDec * self)
{
  self->plugin_data = NULL;     /* We have no internal data */
  return TRUE;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static void
dv_exit (GstTensorDec * self)
{
  /* Nothing to do */
  return;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static gboolean
dv_setOption (GstTensorDec * self, int opNum, const gchar * param)
{
  /* We do not accept anything. */
  return TRUE;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static GstCaps *
dv_getOutputDim (GstTensorDec * self, const GstTensorConfig * config)
{
  /* Old gst_tensordec_video_caps_from_config () had this */
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

/** @brief get video output buffer size */
static gsize
_get_video_xraw_bufsize (tensor_dim dim)
{
  /* dim[0] is bpp and there is zeropadding only when dim[0]%4 > 0 */
  return ((dim[0] * dim[1] - 1) / 4 + 1) * 4 * dim[2];
}

/** @brief tensordec-plugin's TensorDecDef callback */
static gsize
dv_getTransformSize (GstTensorDec * self, GstCaps * caps,
    gsize size, GstCaps * othercaps, GstPadDirection direction)
{
  GstTensorConfig *config = &self->tensor_config;
  uint32_t *dim = &(config->info.dimension[0]);

  if (direction == GST_PAD_SINK)
    return _get_video_xraw_bufsize (dim);
  else
    return 0; /** @todo NYI */
}

/** @brief tensordec-plugin's TensorDecDef callback */
static GstFlowReturn
dv_decode (GstTensorDec * self, const GstTensorMemory * input,
    GstBuffer * outbuf)
{
  GstMapInfo out_info;
  GstMemory *out_mem;
  GstTensorConfig *config = &self->tensor_config;
  uint32_t *dim = &(config->info.dimension[0]);
  size_t size = _get_video_xraw_bufsize (dim);

  g_assert (outbuf);
  if (gst_buffer_get_size (outbuf) > 0 && gst_buffer_get_size (outbuf) != size) {
    gst_buffer_set_size (outbuf, size);
  }
  g_assert (config->info.type == _NNS_UINT8);

  if (gst_buffer_get_size (outbuf) == size) {
    /* Don't reallocate. Reuse what's already given */
    out_mem = gst_buffer_get_all_memory (outbuf);
  } else {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  }
  g_assert (gst_memory_map (out_mem, &out_info, GST_MAP_WRITE));

  if (0 == ((dim[0] * dim[1]) % 4)) {
    /* No Padding Required */
    memcpy (out_info.data, input->data, input->size);
  } else {
    /* Do Padding */
    int h;
    uint8_t *ptr, *inp;

    ptr = (uint8_t *) out_info.data;
    inp = (uint8_t *) input->data;
    for (h = 0; h < dim[2]; h++) {
      memcpy (ptr, inp, dim[0] * dim[1]);
      inp += (dim[0] * dim[1]);
      ptr += ((dim[0] * dim[1] - 1) / 4 + 1) * 4;
    }
  }
  gst_memory_unmap (out_mem, &out_info);

  if (gst_buffer_get_size (outbuf) == 0)
    gst_buffer_append_memory (outbuf, out_mem);

  /** @todo Caller of dv_decode in tensordec.c should call gst_memory_unmap to inbuf */

  return GST_FLOW_OK;
}

/** @brief Direct-Video tensordec-plugin TensorDecDef instance */
static TensorDecDef directVideo = {
  .modename = "direct_video",
  .type = OUTPUT_VIDEO,
  .init = dv_init,
  .exit = dv_exit,
  .setOption = dv_setOption,
  .getOutputDim = dv_getOutputDim,
  .getTransformSize = dv_getTransformSize,
  .decode = dv_decode,
};

/** @brief Initialize this object for tensordec-plugin */
__attribute__ ((constructor))
     void init (void)
{
  tensordec_probe (&directVideo);
}

/** @brief Destruct this object for tensordec-plugin */
__attribute__ ((destructor))
     void fini (void)
{
  tensordec_exit (directVideo.modename);
}
