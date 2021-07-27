/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor_converter subplugin, "protobuf"
 * Copyright (C) 2020 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file        tensor_converter_protobuf.cc
 * @date        2 June 2020
 * @brief       NNStreamer tensor-converter subplugin, "protobuf",
 *              which converts protobuf byte stream to tensors.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

/**
 * Install protobuf
 * We assume that you use Ubuntu linux distribution.
 * You may simply download binary packages from PPA
 *
 * $ sudo apt-add-repository ppa:nnstreamer
 * $ sudo apt update
 * $ sudo apt install libprotobuf-dev libprotobuf-lite17 libprotobuf17
 * protobuf-compiler17
 */

#include <fstream>
#include <glib.h>
#include <gst/gstinfo.h>
#include <iostream>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <string>
#include <typeinfo>
#include <nnstreamer_plugin_api_converter.h>
#include <nnstreamer_util.h>
#include "nnstreamer_protobuf.h"
#include "tensor_converter_util.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_pbc (void) __attribute__((constructor));
void fini_pbc (void) __attribute__((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */
/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
static GstCaps *
pbc_query_caps (const GstTensorsConfig *config)
{
  UNUSED (config);
  return gst_caps_from_string (GST_PROTOBUF_TENSOR_CAP_DEFAULT);
}

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
static GstBuffer *
pbc_convert (GstBuffer *in_buf, GstTensorsConfig *config, void *priv_data)
{
  UNUSED (priv_data);
  return gst_tensor_converter_protobuf (in_buf, config, NULL);
}

static gchar converter_subplugin_protobuf[] = "protobuf";

/** @brief protobuf tensor converter sub-plugin NNStreamerExternalConverter instance */
static NNStreamerExternalConverter protobuf = {
  .name = converter_subplugin_protobuf,
  .convert = pbc_convert,
  .get_out_config = tcu_get_out_config,
  .query_caps = pbc_query_caps,
  .open = NULL,
  .close = NULL
};

/** @brief Initialize this object for tensor converter sub-plugin */
void
init_pbc (void)
{
  registerExternalConverter (&protobuf);
}

/** @brief Destruct this object for tensor converter sub-plugin */
void
fini_pbc (void)
{
  unregisterExternalConverter (protobuf.name);
}
