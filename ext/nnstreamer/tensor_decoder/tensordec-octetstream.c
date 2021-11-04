/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-Decoder
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	tensordec-octetstream.c
 * @date	04 Nov 2021
 * @brief	NNStreamer tensor-decoder subplugin, "octet stream",
 *              which converts tensors to octet stream.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <string.h>
#include <glib.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "tensordecutil.h"

void init_os (void) __attribute__ ((constructor));
void fini_os (void) __attribute__ ((destructor));

#define OCTET_CAPS_STR "application/octet-stream"

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
os_init (void **pdata)
{
  *pdata = NULL;
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
os_exit (void **pdata)
{
  UNUSED (pdata);
  return;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
os_setOption (void **pdata, int opNum, const char *param)
{
  UNUSED (pdata);
  UNUSED (opNum);
  UNUSED (param);
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
os_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
  GstCaps *caps;
  UNUSED (pdata);

  caps = gst_caps_from_string (OCTET_CAPS_STR);
  setFramerateFromConfig (caps, config);
  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
os_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  guint i;
  gboolean is_flexible;
  GstTensorMetaInfo meta;
  gpointer mem_data;
  UNUSED (pdata);

  if (!config || !input || !outbuf) {
    ml_loge ("NULL parameter is passed to tensor_decoder::octet_stream");
    return GST_FLOW_ERROR;
  }
  is_flexible = gst_tensors_config_is_flexible (config);

  for (i = 0; i < config->info.num_tensors; i++) {
    gsize offset = 0, data_size = 0;
    GstMemory *mem = NULL;

    if (is_flexible) {
      gst_tensor_meta_info_parse_header (&meta, input[i].data);
      offset = gst_tensor_meta_info_get_header_size (&meta);
      data_size = gst_tensor_meta_info_get_data_size (&meta);
    } else {
      data_size = gst_tensors_info_get_size (&config->info, i);
    }
    mem_data = _g_memdup ((guint8 *) input[i].data + offset, data_size);
    mem = gst_memory_new_wrapped ((GstMemoryFlags) 0, mem_data, data_size,
          0, data_size, NULL, g_free);
    gst_buffer_append_memory (outbuf, mem);
  }

  return GST_FLOW_OK;
}

static gchar decoder_subplugin_octet_stream[] = "octet_stream";

/** @brief octet stream tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef octetSTream = {
  .modename = decoder_subplugin_octet_stream,
  .init = os_init,
  .exit = os_exit,
  .setOption = os_setOption,
  .getOutCaps = os_getOutCaps,
  .getTransformSize = NULL,
  .decode = os_decode
};

/** @brief Initialize this object for tensordec-plugin */
void
init_os (void)
{
  nnstreamer_decoder_probe (&octetSTream);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_os (void)
{
  nnstreamer_decoder_exit (octetSTream.modename);
}
