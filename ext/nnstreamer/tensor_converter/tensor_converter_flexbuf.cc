/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor_converter subplugin, "Flexbuffer"
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file        tensor_converter_flexbuf.cc
 * @date        12 Mar 2021
 * @brief       NNStreamer tensor-converter subplugin, "flexbuffer",
 *              which converts flexbuffers byte stream to tensors.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */
/**
 * SECTION:tensor_converter::flexbuf
 * @see https://google.github.io/flatbuffers/flexbuffers.html
 *
 * tensor_converter::flexbuf converts flexbuffers to tensors stream..
 * The output is always in the format of other/tensor or other/tensors.
 *
 * Binary format of the flexbuffers for tensors (default in nnstreamer).
 * Each data is represented in `KEY : TYPE | <VALUE>` form.
 *
 * Map {
 *   "num_tensors" : UInt32 | <The number of tensors>
 *   "rate_n" : Int32 | <Framerate numerator>
 *   "rate_d" : Int32 | <Framerate denominator>
 *   "tensor_#": Vector | { String | <tensor name>,
 *                          Int32 | <data type>,
 *                          Vector | <tensor dimension>,
 *                          Blob | <tensor data>
 *                         }
 * }
 *
 * If you want to convert your own binary format of the flexbuffers to tensors,
 * You can use custom mode of the tensor converter.
 * @code
 * // Define custom callback function
 * GstBuffer * tensor_converter_custom_cb (GstBuffer *in_buf,
 *     GstTensorsConfig *config) {
 *   // Write a code to convert flexbuffers to tensors.
 * }
 *
 * ...
 * // Register custom callback function
 * nnstreamer_converter_custom_register ("tconv", tensor_converter_custom_cb, NULL);
 * ...
 * // Use the custom tensor converter in a pipeline.
 * // E.g., Pipeline of " ... (flexbuffers) ! tensor_converter mode=custom:tconv ! (tensors)... "
 * ...
 * // After everything is done.
 * nnstreamer_converter_custom_unregister ("tconv");
 * @endcode
 */

#include <glib.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_converter.h>
#include <flatbuffers/flexbuffers.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_flxc (void) __attribute__ ((constructor));
void fini_flxc (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
static GstCaps *
flxc_query_caps (const GstTensorsConfig *config)
{
  return gst_caps_from_string (GST_FLEXBUF_CAP_DEFAULT);
}

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
static gboolean
flxc_get_out_config (const GstCaps *in_cap, GstTensorsConfig *config)
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

/** @brief tensor converter plugin's NNStreamerExternalConverter callback
 */
static GstBuffer *
flxc_convert (GstBuffer *in_buf, gsize *frame_size, guint *frames_in, GstTensorsConfig *config)
{
  GstBuffer *out_buf = NULL;
  GstMemory *in_mem, *out_mem;
  GstMapInfo in_info;
  guint mem_size;
  gpointer mem_data;

  in_mem = gst_buffer_peek_memory (in_buf, 0);
  if (gst_memory_map (in_mem, &in_info, GST_MAP_READ) == FALSE) {
    ml_loge ("Cannot map input memory / tensor_converter::flexbuf.\n");
    return NULL;
  }

  flexbuffers::Map tensors = flexbuffers::GetRoot (in_info.data, in_info.size).AsMap ();
  config->info.num_tensors = tensors["num_tensors"].AsUInt32 ();

  if (config->info.num_tensors > NNS_TENSOR_SIZE_LIMIT) {
    nns_loge ("The number of tensors is limited to %d", NNS_TENSOR_SIZE_LIMIT);
    goto done;
  }
  config->rate_n = tensors["rate_n"].AsInt32 ();
  config->rate_d = tensors["rate_d"].AsInt32 ();
  out_buf = gst_buffer_new ();
  *frame_size = 0;
  *frames_in = 1;

  for (guint i = 0; i < config->info.num_tensors; i++) {
    gchar * tensor_key = g_strdup_printf ("tensor_%d", i);
    flexbuffers::Vector tensor = tensors[tensor_key].AsVector ();
    config->info.info[i].name = g_strdup (tensor[0].AsString ().c_str ());
    config->info.info[i].type = (tensor_type) tensor[1].AsInt32 ();

    flexbuffers::TypedVector dim = tensor[2].AsTypedVector ();
    for (guint j = 0; j < NNS_TENSOR_RANK_LIMIT; j++) {
      config->info.info[i].dimension[j] = dim[j].AsInt32 ();
    }
    flexbuffers::Blob tensor_data = tensor[3].AsBlob ();
    mem_size = tensor_data.size ();
    *frame_size += mem_size;

    /** @todo Consider removing memory copies for better performance. */
    mem_data = g_memdup (tensor_data.data (), mem_size);

    out_mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY, mem_data,
        mem_size, 0, mem_size, mem_data, g_free);

    gst_buffer_append_memory (out_buf, out_mem);
    g_free (tensor_key);
  }

  /** copy timestamps */
  gst_buffer_copy_into (
      out_buf, in_buf, (GstBufferCopyFlags)GST_BUFFER_COPY_METADATA, 0, -1);
done:
  gst_memory_unmap (in_mem, &in_info);

  return out_buf;
}

static const gchar converter_subplugin_flexbuf[] = "flexbuf";

/** @brief flexbuffer tensor converter sub-plugin NNStreamerExternalConverter instance */
static NNStreamerExternalConverter flexBuf = {
  .name = converter_subplugin_flexbuf,
  .convert = flxc_convert,
  .get_out_config = flxc_get_out_config,
  .query_caps = flxc_query_caps
};

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/** @brief Initialize this object for tensor converter sub-plugin */
void
init_flxc (void)
{
  registerExternalConverter (&flexBuf);
}

/** @brief Destruct this object for tensor converter sub-plugin */
void
fini_flxc (void)
{
  unregisterExternalConverter (flexBuf.name);
}
#ifdef __cplusplus
}
#endif /* __cplusplus */
