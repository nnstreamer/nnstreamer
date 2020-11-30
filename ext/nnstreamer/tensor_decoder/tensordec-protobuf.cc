/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "protobuf"
 * Copyright (C) 2020 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 */
/**
 * @file        tensordec-protobuf.cc
 * @date        25 Mar 2020
 * @brief       NNStreamer tensor-decoder subplugin, "protobuf",
 *              which converts tensor or tensors to Protocol Buffers.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <tensor_typedef.h>
#include "nnstreamer_protobuf.h"
#include "tensordecutil.h"

void init_pb (void) __attribute__ ((constructor));
void fini_pb (void) __attribute__ ((destructor));

/**
 * @brief tensordec-plugin's GstTensorDecoderDef callback
 */
static int
pb_init (void **pdata)
{
  *pdata = NULL; /* no private data are needed for this sub-plugin */
  return TRUE;
}

/** 
 * @brief tensordec-plugin's GstTensorDecoderDef callback 
 */
static void
pb_exit (void **pdata)
{
  return;
}

/**
 * @brief tensordec-plugin's GstTensorDecoderDef callback
 */
static int
pb_setOption (void **pdata, int opNum, const char *param)
{
  return TRUE;
}

/**
 * @brief tensordec-plugin's GstTensorDecoderDef callback
 */
static GstCaps *
pb_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  GstCaps *caps;
  caps = gst_caps_from_string (GST_PROTOBUF_TENSOR_CAP_DEFAULT);
  setFramerateFromConfig (caps, config);
  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
pb_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  return gst_tensor_decoder_protobuf (config, input, outbuf);
}

static gchar decoder_subplugin_protobuf[] = "protobuf";

/**
 * @brief protocol buffers tensordec-plugin GstTensorDecoderDef instance
 */
static GstTensorDecoderDef protobuf = {.modename = decoder_subplugin_protobuf,
  .init = pb_init,
  .exit = pb_exit,
  .setOption = pb_setOption,
  .getOutCaps = pb_getOutCaps,
  .decode = pb_decode };

/**
 * @brief Initialize this object for tensordec-plugin
 */
void
init_pb (void)
{
  nnstreamer_decoder_probe (&protobuf);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_pb (void)
{
  nnstreamer_decoder_exit (protobuf.modename);
}
