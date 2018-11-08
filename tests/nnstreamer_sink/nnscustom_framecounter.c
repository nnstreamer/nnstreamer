/**
 * "Frame Counter"
 * NNStreamer Custom Filter for Multistream Synchronization Test
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * LICENSE: LGPL-2.1
 *
 * @file  nnscustom_framecounter.c
 * @date  08 Nov 2018
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief  Custom filter that gives you the frame number as 1:1:1:1 uint32 tensor
 *
 * Input: ANY other/tensor or other/tensors
 * Output: 1:1:1:1 uint32 other/tensor
 *
 * If Input dimension is 1:1:1:1 uint32, single-tensor,
 * The output will copy the same value, ignoring the internal counter.
 *
 * @bug  No known bugs
 */

#include <tensor_filter_custom.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

static uint32_t maxid = 0;

/**
 * @brief nnstreamer custom filter private data for Frame Counter
 */
typedef struct _pt_data
{
  uint32_t id;
  uint32_t counter; /***< This counts the frame number from 0 */
  int copy; /***< Set 1 if input is 1:1:1:1 uint32 */
} pt_data;

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));
  maxid = maxid + 1;
  data->id = maxid;
  data->counter = 0U;
  data->copy = 0;
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
  int i;
  pt_data *data = _data;
  out_info->num_tensors = 1;
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    out_info->info[0].dimension[i] = 1;
  out_info->info[0].type = _NNS_UINT32;

  data->copy = 1;
  if (in_info->info[0].type != _NNS_UINT32) {
    data->copy = 0;
  } else {
    for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
      if (in_info->info[0].dimension[i] != 1) {
        data->copy = 0;
        break;
      }
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
  uint32_t *counter = (uint32_t *) output[0].data;

  if (data->copy == 1) {
    uint32_t *incoming;
    incoming = (uint32_t *) input[0].data;
    *counter = *incoming;
  } else {
    *counter = data->counter;
  }
  data->counter = data->counter + 1;
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
