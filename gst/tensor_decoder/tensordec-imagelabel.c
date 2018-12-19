/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "image labeling"
 * Copyright (C) 2018 Jinhyuck Park <jinhyuck83.park@samsung.com>
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
 * @file        tensordec-imagelabel.c
 * @date        14 Nov 2018
 * @brief       NNStreamer tensor-decoder subplugin, "image labeling",
 *              which converts image label tensors to text stream.
 *
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

/** @todo getline requires _GNU_SOURCE. remove this later. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include <gst/gstinfo.h>
#include <tensor_common.h>
#include "tensordec.h"

/** @brief Internal data structure for image labeling */
typedef struct
{
  gchar *label_path; /**< Label file path. */
  gchar **labels; /**< The list of loaded labels. Null if not loaded */
  guint total_labels; /**< The number of loaded labels */
  guint max_word_length; /**< The max size of labels */
} ImageLabelData;

/** @brief tensordec-plugin's TensorDecDef callback */
static gboolean
il_init (GstTensorDec * self)
{
  /** @todo check if we need to ensure plugin_data is not yet allocated */
  self->plugin_data = g_new0 (ImageLabelData, 1);
  return TRUE;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static void
il_exit (GstTensorDec * self)
{
  ImageLabelData *data = self->plugin_data;
  if (data->labels) {
    int i;
    for (i = 0; i < data->total_labels; i++)
      g_free (data->labels[i]);
    g_free (data->labels);
  }
  if (data->label_path)
    g_free (data->label_path);

  g_free (self->plugin_data);
  self->plugin_data = NULL;
}

/**
 * @brief Load label file into the internal data
 * @param[in/out] data The given ImageLabelData struct.
 */
static void
loadImageLabels (ImageLabelData * data)
{
  FILE *fp;
  int i;

  /* Clean up previously configured data first */
  if (data->labels) {
    for (i = 0; i < data->total_labels; i++)
      g_free (data->labels[i]);
    g_free (data->labels);
  }
  data->labels = NULL;
  data->total_labels = 0;
  data->max_word_length = 0;

  if ((fp = fopen (data->label_path, "r")) != NULL) {
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    gchar *label;

    GList *labels = NULL, *cursor;

    while ((read = getline (&line, &len, fp)) != -1) {
      if (line) {
        label = g_strdup ((gchar *) line);
        labels = g_list_append (labels, label);
        free (line);
        if (strlen (label) > data->max_word_length)
          data->max_word_length = strlen (label);
      }
      line = NULL;
      len = 0;
    }

    if (line) {
      free (line);
    }

    /* Flatten labels (GList) into data->labels (array gchar **) */
    data->total_labels = g_list_length (labels);
    data->labels = g_new (gchar *, data->total_labels);
    i = 0;
    cursor = g_list_first (labels);
    for (cursor = labels; cursor != NULL; cursor = cursor->next) {
      data->labels[i] = cursor->data;
      i++;
      g_assert (i <= data->total_labels);
    }
    g_list_free (labels);       /* Do not free its elements */

    fclose (fp);
  } else {
    GST_ERROR ("Cannot load label file %s", data->label_path);
    return;
  }

  GST_INFO ("Loaded image label file successfully. %u labels loaded.",
      data->total_labels);
  return;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static gboolean
il_setOption (GstTensorDec * self, int opNum, const gchar * param)
{
  ImageLabelData *data = self->plugin_data;

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

/** @brief tensordec-plugin's TensorDecDef callback */
static GstCaps *
il_getOutCaps (GstTensorDec * self, const GstTensorsConfig * config)
{
  const uint32_t *dim;
  int i;

  g_return_val_if_fail (config != NULL, NULL);
  g_return_val_if_fail (config->info.num_tensors >= 1, NULL);

  /* Even if it's multi-tensor, we use the first tensor only in image labeling */
  dim = config->info.info[0].dimension;
  /* This allows N:1:1:1 only! */
  for (i = 1; i < NNS_TENSOR_RANK_LIMIT; i++)
    if (dim[i] != 1) {
      GST_ERROR ("Dimention %d is not 1, %u", i, dim[i]);
      return NULL;
    }

  return gst_caps_from_string (GST_TENSOR_TEXT_CAPS_STR);
}

/** @brief tensordec-plugin's TensorDecDef callback */
static gsize
il_getTransformSize (GstTensorDec * self, GstCaps * caps,
    gsize size, GstCaps * othercaps, GstPadDirection direction)
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


/** @brief tensordec-plugin's TensorDecDef callback */
static GstFlowReturn
il_decode (GstTensorDec * self, const GstTensorMemory * input,
    GstBuffer * outbuf)
{
  ImageLabelData *data = self->plugin_data;
  GstMapInfo out_info;
  GstMemory *out_mem;

  gsize bpe = tensor_element_size[self->tensor_config.info.info[0].type];
  tensor_element max_val;
  guint max_index = 0;
  gsize num_data;               /* Size / bpe */
  void *input_data;

  gsize size;
  gchar *str;

  g_assert (bpe > 0);
  g_assert (outbuf);

  input_data = input->data;
  num_data = gst_tensor_info_get_size (&self->tensor_config.info.info[0]) / bpe;

  switch (self->tensor_config.info.info[0].type) {
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

/** @brief Image Labeling tensordec-plugin TensorDecDef instance */
static TensorDecDef imageLabeling = {
  .modename = "image_labeling",
  .type = OUTPUT_TEXT,
  .init = il_init,
  .exit = il_exit,
  .setOption = il_setOption,
  .getOutCaps = il_getOutCaps,
  .getTransformSize = il_getTransformSize,
  .decode = il_decode,
};

/** @brief Initialize this object for tensordec-plugin */
__attribute__ ((constructor))
     void init_il (void)
{
  tensordec_probe (&imageLabeling);
}

/** @brief Destruct this object for tensordec-plugin */
__attribute__ ((destructor))
     void fini_il (void)
{
  tensordec_exit (imageLabeling.modename);
}
