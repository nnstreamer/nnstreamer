/**
 * NNStreamer Custom Filter Example 1. Pass-Through
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * LICENSE: Apache-2.0
 *
 * @file	nnstreamer_customfilter_example_passthrough.c
 * @date	11 Jun 2018
 * @brief	Custom NNStreamer Filter Example 1. "Pass-Through"
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * As a test, before #71 is resolved, this will support "3x280x40" uint8 tensors.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <tensor_filter_custom.h>

#define D1	(3)
#define D2	(280)
#define D3	(40)

typedef struct _pt_data
{
  uint32_t id; /***< Just for testing */
  uint32_t dim[NNS_TENSOR_RANK_LIMIT];
  tensor_type type;
} pt_data;

static void *
pt_init (const GstTensor_Filter_Properties * prop)
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

static void
pt_exit (void *private_data, const GstTensor_Filter_Properties * prop)
{
  pt_data *data = private_data;
  g_assert (data);
  free (data);
}

static int
get_inputDim (void *private_data, const GstTensor_Filter_Properties * prop,
    uint32_t inputDimension[NNS_TENSOR_RANK_LIMIT], tensor_type * type)
{
  pt_data *data = private_data;
  int i;

  g_assert (data);
  g_assert (NNS_TENSOR_RANK_LIMIT >= 3);
  inputDimension[0] = D1;
  inputDimension[1] = D2;
  inputDimension[2] = D3;
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    inputDimension[i] = 1;
  *type = _NNS_UINT8;
  return 0;
  return 0;
}

static int
get_outputDim (void *private_data, const GstTensor_Filter_Properties * prop,
    uint32_t outputDimension[NNS_TENSOR_RANK_LIMIT], tensor_type * type)
{
  pt_data *data = private_data;
  int i;

  g_assert (data);
  g_assert (NNS_TENSOR_RANK_LIMIT >= 3);
  outputDimension[0] = D1;
  outputDimension[1] = D2;
  outputDimension[2] = D3;
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    outputDimension[i] = 1;
  *type = _NNS_UINT8;
  return 0;
}

static int
pt_invoke (void *private_data, const GstTensor_Filter_Properties * prop,
    uint8_t * inptr, uint8_t * outptr)
{
  pt_data *data = private_data;
  size_t size;

  g_assert (data);
  g_assert (inptr);
  g_assert (outptr);

  size = get_tensor_element_count (data->dim) * tensor_element_size[data->type];

  g_assert (inptr != outptr);
  memcpy (outptr, inptr, size);

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
