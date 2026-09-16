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

/**
 * @brief Check that the data a serialized stream carries for a tensor holds what the stream declares for it.
 */
gboolean
tcu_check_tensor_data (const GstTensorsConfig * config,
    const GstTensorInfo * info, const guint8 * data, gsize size)
{
  GstTensorMetaInfo meta;
  gsize hsize, expected;

  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (info != NULL, FALSE);

  if (config->info.format >= _NNS_TENSOR_FORMAT_END) {
    ml_loge ("The stream declares an unknown tensor format %d.",
        config->info.format);
    return FALSE;
  }

  if (gst_tensors_config_is_static (config)) {
    if ((guint) info->type >= _NNS_END || !gst_tensor_info_validate (info)) {
      ml_loge ("The stream declares an invalid type or dimension of a tensor.");
      return FALSE;
    }

    expected = gst_tensor_info_get_size (info);
    if (size != expected) {
      ml_loge
          ("The stream carries %zu bytes for a tensor declared as %zu bytes.",
          size, expected);
      return FALSE;
    }

    return TRUE;
  }

  gst_tensor_meta_info_init (&meta);
  if (!data || size < gst_tensor_meta_info_get_header_size (&meta) ||
      !gst_tensor_meta_info_parse_header (&meta, (gpointer) data)) {
    ml_loge ("The stream carries a tensor without a valid meta header.");
    return FALSE;
  }

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  expected = hsize + gst_tensor_meta_info_get_data_size (&meta);
  if (hsize == 0 || size < expected) {
    ml_loge
        ("The stream carries %zu bytes for a tensor whose meta header declares %zu bytes.",
        size, expected);
    return FALSE;
  }

  return TRUE;
}
