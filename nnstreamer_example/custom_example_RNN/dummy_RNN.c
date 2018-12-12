/**
 * NNStreamer Custom Filter RNN Example, "dummyRNN"
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * LICENSE: LGPL-2.1
 *
 * @file	dummy_RNN.c
 * @date	02 Nov 2018
 * @brief	Custom NNStreamer RNN Model. "Dummy RNN"
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This supports two "4:4:4:1" uint8 tensors, where the first one is the new tensor and the second one is "recurring" tensor (output of the previous frame).
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <tensor_filter_custom.h>

#define TSIZE   (4)

/**
 * @brief _pt_data
 */
typedef struct _pt_data
{
  uint32_t id; /***< Just for testing */
  uint32_t counter; /**< Internal frame counter for debugging/demo */
  GstTensorInfo info[2]; /**< tensor info. 0:new frame / 1:recurring frame */
} pt_data;

/**
 * @brief Initialize dummy-RNN
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));
  int i;

  data->id = 0;
  data->counter = 0;
  data->info[0].dimension[0] = TSIZE;
  data->info[0].dimension[1] = TSIZE;
  data->info[0].dimension[2] = TSIZE;
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    data->info[0].dimension[i] = 1;
  data->info[0].type = _NNS_UINT8;
  data->info[1] = data->info[0];

  return data;
}

/**
 * @brief Exit dummy-RNN
 */
static void
pt_exit (void *private_data, const GstTensorFilterProperties * prop)
{
  pt_data *data = private_data;
  assert (data);
  free (data);
}

/**
 * @brief get the input tensor dimensions of dummy-RNN (4:4:4 uint8, 4:4:4 uint8)
 */
static int
get_inputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  pt_data *data = private_data;

  assert (data);
  assert (NNS_TENSOR_RANK_LIMIT >= 3);

  info->num_tensors = 2;
  info->info[0] = data->info[0];
  info->info[1] = data->info[1];
  return 0;
}

/**
 * @brief get the output tensor dimensions of dummy-RNN (4:4:4 uint8)
 */
static int
get_outputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  pt_data *data = private_data;

  assert (data);
  assert (NNS_TENSOR_RANK_LIMIT >= 3);

  info->num_tensors = 1;
  info->info[0] = data->info[0];
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
  uint8_t *in0, *in1, *out;

  if (!data || !input || !output)
    return -EINVAL;

  in0 = input[0].data;
  in1 = input[1].data;
  out = output[0].data;

  if (!in0 || !in1 || !out)
    return -EINVAL;

  for (h = 0; h < 4; h++) {
    w = 0;
    memcpy (&(out[location (0, w, h)]), &(in0[location (0, w, h)]), TSIZE);
    for (w = 1; w <= 2; w++) {
      for (c = 0; c < TSIZE; c++) {
        uint16_t sum = in0[location (0, w, h)];
        sum += in1[location (0, w, h)];
        out[location (0, w, h)] = sum / 2;
      }
    }
    w = 3;
    memcpy (&(out[location (0, w, h)]), &(in1[location (0, w, h)]), TSIZE);
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
