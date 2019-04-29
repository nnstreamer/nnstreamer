/**
 * NNStreamer Custom Filter Example 2. Pass-Through with Variable Dimensions
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * LICENSE: LGPL-2.1
 *
 * @file	nnstreamer_customfilter_example_passthrough_variable.c
 * @date	22 Jun 2018
 * @brief	Custom NNStreamer Filter Example 2. "Pass-Through with Variable Dimensions"
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api.h>

/**
 * @brief _pt_data
 */
typedef struct _pt_data
{
  uint32_t id; /***< Just for testing */
} pt_data;

/**
 * @brief pt_init
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));
  assert (data);

  data->id = 0;
  return data;
}

/**
 * @brief pt_exit
 */
static void
pt_exit (void *private_data, const GstTensorFilterProperties * prop)
{
  pt_data *data = private_data;
  assert (data);
  free (data);
}

/**
 * @brief set_inputDim
 */
static int
set_inputDim (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  int i, t;

  assert (in_info);
  assert (out_info);

  out_info->num_tensors = in_info->num_tensors;

  for (t = 0; t < in_info->num_tensors; t++) {
    for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
      out_info->info[t].dimension[i] = in_info->info[t].dimension[i];
    }

    out_info->info[t].type = in_info->info[t].type;
  }

  return 0;
}

/**
 * @brief pt_invoke
 */
static int
pt_invoke (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  pt_data *data = private_data;
  size_t size;
  int t;

  assert (data);
  assert (input);
  assert (output);

  for (t = 0; t < prop->output_meta.num_tensors; t++) {
    size = gst_tensor_info_get_size (&prop->output_meta.info[t]);

    assert (input[t].data != output[t].data);
    memcpy (output[t].data, input[t].data, size);
  }

  return 0;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = pt_init,
  .exitfunc = pt_exit,
  .setInputDim = set_inputDim,
  .invoke = pt_invoke,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
