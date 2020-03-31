/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "image labeling"
 * Copyright (C) 2018 Jinhyuck Park <jinhyuck83.park@samsung.com>
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
 * @file        tensordec-imagelabel.c
 * @date        14 Nov 2018
 * @brief       NNStreamer tensor-decoder subplugin, "image labeling",
 *              which converts image label tensors to text stream.
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include <gst/gstinfo.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api.h>

void init_il (void) __attribute__ ((constructor));
void fini_il (void) __attribute__ ((destructor));

#define DECODER_IL_TEXT_CAPS_STR \
    "text/x-raw, format = (string) utf8"

/** @brief Internal data structure for image labeling */
typedef struct
{
  char *label_path; /**< Label file path. */
  char **labels; /**< The list of loaded labels. Null if not loaded */
  guint total_labels; /**< The number of loaded labels */
  guint max_word_length; /**< The max size of labels */
} ImageLabelData;

/**
 * @brief Free the allocated lables
 */
static void
_free_labels (ImageLabelData * data)
{
  guint i;

  if (data->labels) {
    for (i = 0; i < data->total_labels; i++)
      g_free (data->labels[i]);
    g_free (data->labels);
  }

  data->labels = NULL;
  data->total_labels = 0;
  data->max_word_length = 0;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
il_init (void **pdata)
{
  /** @todo check if we need to ensure plugin_data is not yet allocated */
  *pdata = g_new0 (ImageLabelData, 1);
  if (*pdata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
il_exit (void **pdata)
{
  ImageLabelData *data = *pdata;

  _free_labels (data);

  if (data->label_path)
    g_free (data->label_path);

  g_free (*pdata);
  *pdata = NULL;
}

/**
 * @brief Load label file into the internal data
 * @param[in/out] data The given ImageLabelData struct.
 */
static void
loadImageLabels (ImageLabelData * data)
{
  GError *err = NULL;
  gchar **labels;
  gchar *contents = NULL;
  guint i, len;

  /* Clean up previously configured data first */
  _free_labels (data);

  /* Read file contents */
  if (!g_file_get_contents (data->label_path, &contents, NULL, &err)) {
    GST_ERROR ("Unable to read file %s with error %s.",
        data->label_path, err->message);
    g_clear_error (&err);
    return;
  }

  labels = g_strsplit (contents, "\n", -1);
  data->total_labels = g_strv_length (labels);
  data->labels = g_new0 (char *, data->total_labels);
  if (data->labels == NULL) {
    GST_ERROR ("Failed to allocate memory for label data.");
    data->total_labels = 0;
    goto error;
  }

  for (i = 0; i < data->total_labels; i++) {
    data->labels[i] = g_strdup (labels[i]);

    len = strlen (labels[i]);
    if (len > data->max_word_length) {
      data->max_word_length = len;
    }
  }

error:
  g_strfreev (labels);
  g_free (contents);

  if (data->labels != NULL) {
    GST_INFO ("Loaded image label file successfully. %u labels loaded.",
        data->total_labels);
  }
  return;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
il_setOption (void **pdata, int opNum, const char *param)
{
  ImageLabelData *data = *pdata;

  /* opNum 1 = label file path */
  if (opNum == 0) {
    if (NULL != data->label_path)
      g_free (data->label_path);
    data->label_path = g_strdup (param);

    if (NULL != data->label_path)
      loadImageLabels (data);

    if (data->total_labels > 0)
      return TRUE;
    else
      return FALSE;
  }

  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
il_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
  const uint32_t *dim;
  int i;

  g_return_val_if_fail (config != NULL, NULL);
  g_return_val_if_fail (config->info.num_tensors >= 1, NULL);

  /* Even if it's multi-tensor, we use the first tensor only in image labeling */
  dim = config->info.info[0].dimension;
  /* This allows N:1 only! */
  g_return_val_if_fail (dim[0] > 0 && dim[1] == 1, NULL);
  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim[i] == 1, NULL);

  return gst_caps_from_string (DECODER_IL_TEXT_CAPS_STR);
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
il_getTransformSize (void **pdata, const GstTensorsConfig * config,
    GstCaps * caps, size_t size, GstCaps * othercaps, GstPadDirection direction)
{
  return 0;
  /** @todo Use max_word_length if that's appropriate */
}

/** @brief Search for max. Macro for tensor_element union */
#define search_max(type, i, max_index, max_val, bpe, data, num_data) \
do {\
  int i;\
  type *cursor = (type *) data;\
  max_val = cursor[0];\
  max_index = 0;\
  for (i = 1; i < num_data; i++) {\
    if (cursor[i] > max_val) {\
      max_val = cursor[i];\
      max_index = i;\
    }\
  }\
} while (0);

/** @brief Shorter case statement for search_max */
#define search_max_case(type, typename) \
case typename:\
  search_max(type, i, max_index, max_val._##type, bpe, input_data, num_data);\
  break;


/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
il_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  ImageLabelData *data = *pdata;
  GstMapInfo out_info;
  GstMemory *out_mem;

  gsize bpe = gst_tensor_get_element_size (config->info.info[0].type);
  tensor_element max_val;
  guint max_index = 0;
  gsize num_data;               /* Size / bpe */
  void *input_data;

  gsize size;
  char *str;

  g_assert (bpe > 0);
  g_assert (outbuf);

  input_data = input->data;
  num_data = gst_tensor_info_get_size (&config->info.info[0]) / bpe;

  switch (config->info.info[0].type) {
      search_max_case (int32_t, _NNS_INT32);
      search_max_case (uint32_t, _NNS_UINT32);
      search_max_case (int16_t, _NNS_INT16);
      search_max_case (uint16_t, _NNS_UINT16);
      search_max_case (int8_t, _NNS_INT8);
      search_max_case (uint8_t, _NNS_UINT8);
      search_max_case (double, _NNS_FLOAT64);
      search_max_case (float, _NNS_FLOAT32);
      search_max_case (int64_t, _NNS_INT64);
      search_max_case (uint64_t, _NNS_UINT64);
    default:
      return GST_FLOW_NOT_SUPPORTED;
  }

  g_assert (max_index < data->total_labels);

  /** @todo With option-2, allow to change output format */
  str = g_strdup_printf ("%s", data->labels[max_index]);
  size = strlen (str);

  /* Ensure we have outbuf properly allocated */
  if (gst_buffer_get_size (outbuf) == 0) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  g_assert (gst_memory_map (out_mem, &out_info, GST_MAP_WRITE));

  memcpy (out_info.data, str, size);

  gst_memory_unmap (out_mem, &out_info);

  if (gst_buffer_get_size (outbuf) == 0)
    gst_buffer_append_memory (outbuf, out_mem);

  g_free (str);

  return GST_FLOW_OK;
}

static gchar decoder_subplugin_image_labeling[] = "image_labeling";

/** @brief Image Labeling tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef imageLabeling = {
  .modename = decoder_subplugin_image_labeling,
  .init = il_init,
  .exit = il_exit,
  .setOption = il_setOption,
  .getOutCaps = il_getOutCaps,
  .getTransformSize = il_getTransformSize,
  .decode = il_decode
};

/** @brief Initialize this object for tensordec-plugin */
void
init_il (void)
{
  nnstreamer_decoder_probe (&imageLabeling);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_il (void)
{
  nnstreamer_decoder_exit (imageLabeling.modename);
}
