/**
 * NNStreamer Custom Filter Example 1. Pass-Through
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * LICENSE: LGPL-2.1
 *
 * @file	nnstreamer_customfilter_example_passthrough.c
 * @date	11 Jun 2018
 * @brief	Custom NNStreamer Filter Example 1. "Pass-Through"
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * this will supports "3x280x40" uint8 tensors (hardcoded dimensions)
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <tensor_filter_custom.h>
#include <tensor_common.h>

#define D1	(3)
#define D2	(280)
#define D3	(40)

/**
 * @brief _pt_data
 */
typedef struct _pt_data
{
  uint32_t id; /***< Just for testing */
  tensor_dim dim;
  tensor_type type;
} pt_data;

/**
 * @brief _pt_data
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));
  int i;

  data->id = 0;
  data->dim[0] = D1;
  data->dim[1] = D2;
  data->dim[2] = D3;
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    data->dim[i] = 1;
  data->type = _NNS_UINT8;

  return data;
}

/**
 * @brief _pt_data
 */
static void
pt_exit (void *private_data, const GstTensorFilterProperties * prop)
{
  pt_data *data = private_data;
  g_assert (data);
  free (data);
}

/**
 * @brief _pt_data
 */
static int
get_inputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  pt_data *data = private_data;
  int i;

  g_assert (data);
  g_assert (NNS_TENSOR_RANK_LIMIT >= 3);

  info->info[0].dimension[0] = D1;
  info->info[0].dimension[1] = D2;
  info->info[0].dimension[2] = D3;
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    info->info[0].dimension[i] = 1;
  info->info[0].type = _NNS_UINT8;
  info->num_tensors = 1;
  return 0;
}

/**
 * @brief _pt_data
 */
static int
get_outputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  pt_data *data = private_data;
  int i;

  g_assert (data);
  g_assert (NNS_TENSOR_RANK_LIMIT >= 3);

  info->info[0].dimension[0] = D1;
  info->info[0].dimension[1] = D2;
  info->info[0].dimension[2] = D3;
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    info->info[0].dimension[i] = 1;
  info->info[0].type = _NNS_UINT8;
  info->num_tensors = 1;
  return 0;
}

/**
 * @brief _pt_data
 */
static int
pt_invoke (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  pt_data *data = private_data;
  size_t size;

  g_assert (data);
  g_assert (input);
  g_assert (output);

  size = get_tensor_element_count (data->dim) * tensor_element_size[data->type];

  g_assert (input[0].data != output[0].data);
  memcpy (output[0].data, input[0].data, size);

  return 0;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = pt_init,
  .exitfunc = pt_exit,
  .getInputDim = get_inputDim,
  .getOutputDim = get_outputDim,
  .invoke = pt_invoke,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
