/* SPDX-License-Identifier: LGPL-2.1-only */
/**
* @file        tensor_converter_util.h
* @date        26 May 2021
* @brief       Utility functions for NNStreamer tensor-converter subplugins.
* @see         https://github.com/nnstreamer/nnstreamer
* @author      MyungJoo Ham <myungjoo.ham@samsung.com>
* @bug         No known bugs except for NYI items
*/

#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include "tensor_converter_util.h"

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
gboolean
tcu_get_out_config (const GstCaps * in_cap, GstTensorsConfig * config)
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
  if (gst_tensor_parse_dimension ("1:1:1:1",
          config->info.info[0].dimension) == 0) {
    ml_loge ("Failed to set initial dimension for subplugin");
    return FALSE;
  }

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n,
        &config->rate_d);
  } else {
    /* cannot get the framerate */
    config->rate_n = 0;
    config->rate_d = 1;
  }
  return TRUE;
}
