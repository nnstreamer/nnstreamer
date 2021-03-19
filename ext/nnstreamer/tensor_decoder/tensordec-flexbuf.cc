/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "Flexbuffer"
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file        tensordec-flexbuf.cc
 * @date        12 Mar 2021
 * @brief       NNStreamer tensor-decoder subplugin, "flexbuffer",
 *              which converts tensor or tensors to flexbuffer byte stream.
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */
/**
 * SECTION:tensor_decoder::flexbuf
 * @see https://google.github.io/flatbuffers/flexbuffers.html
 *
 * tensor_decoder::flexbuf converts tensors stream to flexbuffers.
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
 * If you want to convert tensors to your own binary format of the flexbuffers,
 * You can use custom mode of the tensor decoder.
 * This is an example of a callback type custom mode.
 * @code
 * // Define custom callback function
 * int * tensor_decoder_custom_cb (const GstTensorMemory *input,
 *   const GstTensorsConfig *config, void * data, GstBuffer * out_buf) {
 *   // Write a code to convert tensors to flexbuffers.
 * }
 *
 * ...
 * // Register custom callback function
 * nnstreamer_decoder_custom_register ("tdec", tensor_converter_custom_cb, NULL);
 * ...
 * // Use the custom tensor converter in a pipeline.
 * // E.g., Pipeline of " ... (tensors) ! tensor_decoder mode=custom-code option1=tdec ! (flexbuffers)... "
 * ...
 * // After everything is done.
 * nnstreamer_decoder_custom_unregister ("tdec");
 * @endcode
 */


#include <glib.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include "tensordecutil.h"
#include <flatbuffers/flexbuffers.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_flxb (void) __attribute__ ((constructor));
void fini_flxb (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
flxb_init (void **pdata)
{
  *pdata = NULL;
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
flxb_exit (void **pdata)
{
  return;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
flxb_setOption (void **pdata, int opNum, const char *param)
{
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
flxb_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  GstCaps *caps;
  caps = gst_caps_from_string (GST_FLEXBUF_CAP_DEFAULT);
  setFramerateFromConfig (caps, config);
  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
flxb_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  GstMapInfo out_info;
  GstMemory *out_mem;
  unsigned int i, num_tensors = config->info.num_tensors;
  gboolean need_alloc;
  size_t flex_size;
  flexbuffers::Builder fbb;

  fbb.Map ([&]() {
    fbb.UInt ("num_tensors", num_tensors);
    fbb.Int ("rate_n", config->rate_n);
    fbb.Int ("rate_d", config->rate_d);
    for (i = 0; i < num_tensors; i++) {
      gchar * tensor_key = g_strdup_printf ("tensor_%d", i);
      gchar * tensor_name = NULL;

      if (config->info.info[i].name == NULL) {
        tensor_name = g_strdup ("");
      } else {
        tensor_name = g_strdup (config->info.info[i].name);
      }
      tensor_type type = config->info.info[i].type;

      fbb.Vector (tensor_key, [&]() {
        fbb += tensor_name;
        fbb += type;
        fbb.Vector (config->info.info[i].dimension, NNS_TENSOR_RANK_LIMIT);
        fbb.Blob (input[i].data, (size_t) input[i].size);
      });
      g_free (tensor_key);
      g_free (tensor_name);
    }
  });
  fbb.Finish ();
  flex_size = fbb.GetSize ();

  g_assert (outbuf);
  need_alloc = (gst_buffer_get_size (outbuf) == 0);

  if (need_alloc) {
    out_mem = gst_allocator_alloc (NULL, flex_size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < flex_size) {
      gst_buffer_set_size (outbuf, flex_size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }

  if (FALSE == gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    if (need_alloc)
      gst_allocator_free (NULL, out_mem);
    nns_loge ("Cannot map gst memory (tensor decoder flexbuf)\n");
    return GST_FLOW_ERROR;
  }

  memcpy (out_info.data, fbb.GetBuffer ().data (), flex_size);

  gst_memory_unmap (out_mem, &out_info);

  if (need_alloc)
    gst_buffer_append_memory (outbuf, out_mem);

  return GST_FLOW_OK;
}

static gchar decoder_subplugin_flexbuf[] = "flexbuf";

/** @brief flexbuffer tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef flexBuf = {.modename = decoder_subplugin_flexbuf,
  .init = flxb_init,
  .exit = flxb_exit,
  .setOption = flxb_setOption,
  .getOutCaps = flxb_getOutCaps,
  .decode = flxb_decode };

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** @brief Initialize this object for tensordec-plugin */
void
init_flxb (void)
{
  nnstreamer_decoder_probe (&flexBuf);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_flxb (void)
{
  nnstreamer_decoder_exit (flexBuf.modename);
}
#ifdef __cplusplus
}
#endif /* __cplusplus */
