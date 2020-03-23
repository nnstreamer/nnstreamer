/**
 * NNStreamer Custom Filter LSTM Example, "dummyLSTM"
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file	dummy_LSTM.c
 * @date	28 Nov 2018
 * @brief	Custom NNStreamer LSTM Model. "Dummy LSTM"
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This supports two "4:4:4:1" float32 tensors, where the first one is the new tensor and the second one is "recurring" tensor (output of the previous frame).
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <tensor_filter_custom.h>

#define TSIZE   (4)

/**
 * @brief _pt_data
 */
typedef struct _pt_data
{
  uint32_t id; /***< Just for testing */
  uint32_t counter; /**< Internal frame counter for debugging/demo */
  GstTensorInfo info[3]; /**< tensor info. 0:new frame / 1:recurring frame 2:recurring frame*/
} pt_data;

/**
 * @brief Initialize dummy-LSTM
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));
  int i;

  assert (data);
  memset (data, 0, sizeof (pt_data));

  data->id = 0;
  data->counter = 0;
  data->info[0].dimension[0] = TSIZE;
  data->info[0].dimension[1] = TSIZE;
  data->info[0].dimension[2] = TSIZE;
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    data->info[0].dimension[i] = 1;
  data->info[0].type = _NNS_FLOAT32;
  data->info[1] = data->info[0];
  data->info[2] = data->info[0];

  return data;
}

/**
 * @brief Exit dummy-LSTM
 */
static void
pt_exit (void *private_data, const GstTensorFilterProperties * prop)
{
  pt_data *data = private_data;
  assert (data);
  free (data);
}

/**
 * @brief get the input tensor dimensions of dummy-LSTM (4:4:4 float32, 4:4:4 float32)
 */
static int
get_inputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  pt_data *data = private_data;

  assert (data);
  assert (NNS_TENSOR_RANK_LIMIT >= 3);

  info->num_tensors = 3;
  /** @todo use common function to copy tensor info */
  info->info[0] = data->info[0];
  info->info[1] = data->info[1];
  info->info[2] = data->info[2];
  return 0;
}

/**
 * @brief get the output tensor dimensions of dummy-LSTM (4:4:4 float32)
 */
static int
get_outputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  pt_data *data = private_data;

  assert (data);
  assert (NNS_TENSOR_RANK_LIMIT >= 3);

  info->num_tensors = 2;
  /** @todo use common function to copy tensor info */
  info->info[0] = data->info[0];
  info->info[1] = data->info[1];
  return 0;
}

/**
 * @brief Get the offset of the tensor data blob pointer.
 */
static size_t
location (uint32_t c, uint32_t w, uint32_t h)
{
  return c + TSIZE * (w + TSIZE * h);
}

/**
 * @brief INFERENCE!
 */
static int
pt_invoke (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  pt_data *data = private_data;
  uint32_t c, w, h;
  float *in0, *in1, *in2, *out0, *out1;
  float in2_tmp0, in2_tmp1;

  if (!data || !input || !output)
    return -EINVAL;

  in0 = input[0].data;
  in1 = input[1].data;
  in2 = input[2].data;
  out0 = output[0].data;
  out1 = output[1].data;

  if (!in0 || !in1 || !in2 || !out0 || !out1)
    return -EINVAL;

  for (h = 0; h < TSIZE; h++) {
    for (w = 0; w < TSIZE; w++) {
      for (c = 0; c < TSIZE; c++) {
        in2_tmp0 = (in2[location (c, w, h)] + in1[location (c, w, h)]) / 2;
        in2_tmp1 = tanh (in2[location (c, w, h)]);
        in0[location (c, w, h)] = in0[location (c, w, h)] * in2_tmp0;
        in0[location (c, w, h)] += (in2_tmp0 * in2_tmp1);
        out0[location (c, w, h)] = in0[location (c, w, h)];
        out1[location (c, w, h)] = tanh (in0[location (c, w, h)]) * in2_tmp0;
      }
    }
  }

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
