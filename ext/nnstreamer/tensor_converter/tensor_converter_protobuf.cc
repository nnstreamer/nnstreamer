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
#include "nnstreamer_plugin_api_converter.h"
#include "nnstreamer_protobuf.h"

void init_pbc (void) __attribute__ ((constructor));
void fini_pbc (void) __attribute__ ((destructor));

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
static GstCaps *
pbc_query_caps (const GstTensorsConfig *config)
{
  return gst_caps_from_string (GST_PROTOBUF_TENSOR_CAP_DEFAULT);
}

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
static gboolean
pbc_get_out_config (const GstCaps *in_cap, GstTensorsConfig *config)
{
  GstStructure *structure;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensors_config_init (config);
  g_return_val_if_fail (in_cap != NULL, FALSE);

  structure = gst_caps_get_structure (in_cap, 0);
  g_return_val_if_fail (structure != NULL, FALSE);

  /* All tensor info should be updated later in chain function. */
  config->info.info[0].type = _NNS_UINT8;
  config->info.num_tensors = 1;
  if (gst_tensor_parse_dimension ("1:1:1:1", config->info.info[0].dimension) == 0) {
    ml_loge ("Failed to set initial dimension for subplugin");
    return FALSE;
  }

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n, &config->rate_d);
  } else {
    /* cannot get the framerate */
    config->rate_n = 0;
    config->rate_d = 1;
  }
  return TRUE;
}

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
static GstBuffer *
pbc_convert (GstBuffer *in_buf, gsize *frame_size, guint *frames_in, GstTensorsConfig *config)
{
  return gst_tensor_converter_protobuf (in_buf, frame_size, frames_in, config);
}

static gchar converter_subplugin_protobuf[] = "libnnstreamer_converter_protobuf";

/** @brief protobuf tensor converter sub-plugin NNStreamerExternalConverter instance */
static NNStreamerExternalConverter protobuf = {.name = converter_subplugin_protobuf,
  .convert = pbc_convert,
  .get_out_config = pbc_get_out_config,
  .query_caps = pbc_query_caps };

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
