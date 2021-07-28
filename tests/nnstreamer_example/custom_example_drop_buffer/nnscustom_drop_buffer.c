/**
 * NNStreamer custom filter to test buffer-drop
 * Copyright (C) 2018 Jaeyun Jung <jy1210.jung@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnscustom_drop_buffer.c
 * @date  25 Jan 2019
 * @author  Jaeyun Jung <jy1210.jung@samsung.com>
 * @brief  Custom filter to drop incoming buffer (skip 9 buffers, then pass 1 buffer)
 * @bug  No known bugs
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <tensor_filter_custom.h>

/**
 * @brief nnstreamer custom filter private data
 */
typedef struct _pt_data
{
  unsigned int count; /**< This counts incoming buffer */
} pt_data;

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));

  assert (data);
  data->count = 0;

  return data;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void
pt_exit (void *_data, const GstTensorFilterProperties * prop)
{
  pt_data *data = _data;

  assert (data);
  free (data);
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
set_inputDim (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  int i, t;

  assert (in_info);
  assert (out_info);

  /** @todo use common function to copy tensor info */
  out_info->num_tensors = in_info->num_tensors;

  for (t = 0; t < in_info->num_tensors; t++) {
    out_info->info[t].name = NULL;
    out_info->info[t].type = in_info->info[t].type;

    for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
      out_info->info[t].dimension[i] = in_info->info[t].dimension[i];
    }
  }

  return 0;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
invoke (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  pt_data *data = _data;
  int t;

  assert (data);

  data->count++;
  if (data->count % 10) {
    /* drop this buffer */
    /** @todo define enum to indicate status code */
    return 2;
  }

  for (t = 0; t < prop->output_meta.num_tensors; t++) {
    memcpy (output[t].data, input[t].data, input[t].size);
  }

  return 0;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = pt_init,
  .exitfunc = pt_exit,
  .setInputDim = set_inputDim,
  .invoke = invoke,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
