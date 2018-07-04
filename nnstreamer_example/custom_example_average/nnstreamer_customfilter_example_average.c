/**
 * NNStreamer Custom Filter Example 4. Average
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * LICENSE: LGPL-2.1
 *
 * @file  nnstreamer_customfilter_example_average.c
 * @date  02 Jul 2018
 * @brief  Custom NNStreamer Filter Example 4. "Average"
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This scales a tensor of [N][y][x][M] to [N][1][1][M]
 *
 * This heavliy suffers from overflows. Do not use this for real applications!!!!!!
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <tensor_filter_custom.h>

typedef struct _pt_data
{
  uint32_t id; /***< Just for testing */
} pt_data;

static void *
pt_init (const GstTensor_Filter_Properties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));

  data->id = 0;
  return data;
}

static void
pt_exit (void *private_data, const GstTensor_Filter_Properties * prop)
{
  pt_data *data = private_data;
  assert (data);
  free (data);
}

static int
set_inputDim (void *private_data, const GstTensor_Filter_Properties * prop,
    const tensor_dim iDim, const tensor_type iType,
    tensor_dim oDim, tensor_type * oType)
{
  int i;
  pt_data *data = private_data;
  assert (data);

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    oDim[i] = iDim[i];

  /* Update [1] and [2] oDim with new-x, new-y */
  oDim[1] = 1;
  oDim[2] = 1;

  *oType = iType;
  return 0;
}


#define do_avg(type, sumtype) do {\
      sumtype *avg = (sumtype *) malloc(sizeof(sumtype) * prop->inputDimension[0]); \
      type *iptr = (type *) inptr; \
      type *optr = (type *) outptr; \
      for (z = 0; z < prop->inputDimension[3]; z++) { \
        for (y = 0; y < prop->inputDimension[0]; y++) \
          avg[y] = 0; \
        for (y = 0; y < prop->inputDimension[2]; y++) { \
          for (x = 0; x < prop->inputDimension[1]; x++) { \
            for (c = 0; c < prop->inputDimension[0]; c++) { \
              avg[c] += *(iptr + c + x * ix + y * iy + z * iz); \
            } \
          } \
        } \
        for (c = 0; c < prop->inputDimension[0]; c++) { \
          *(optr + c + z * prop->inputDimension[0]) = (type) (avg[c] / xy); \
        } \
      } \
      free(avg); \
  } while (0)

static int
pt_invoke (void *private_data, const GstTensor_Filter_Properties * prop,
    const uint8_t * inptr, uint8_t * outptr)
{
  pt_data *data = private_data;
  uint32_t c, x, y, z;

  unsigned ix = prop->inputDimension[0];
  unsigned iy = prop->inputDimension[0] * prop->inputDimension[1];
  unsigned iz =
      prop->inputDimension[0] * prop->inputDimension[1] *
      prop->inputDimension[2];
  unsigned xy = prop->inputDimension[1] * prop->inputDimension[2];

  assert (data);
  assert (inptr);
  assert (outptr);

  /* This assumes the limit is 4 */
  assert (NNS_TENSOR_RANK_LIMIT == 4);

  assert (prop->inputDimension[0] == prop->outputDimension[0]);
  assert (prop->inputDimension[3] == prop->outputDimension[3]);
  assert (prop->inputType == prop->outputType);

  switch (prop->inputType) {
    case _NNS_INT8:
      do_avg (int8_t, int64_t);
      break;
    case _NNS_INT16:
      do_avg (int16_t, int64_t);
      break;
    case _NNS_INT32:
      do_avg (int32_t, int64_t);
      break;
    case _NNS_UINT8:
      do_avg (uint8_t, uint64_t);
      break;
    case _NNS_UINT16:
      do_avg (uint16_t, uint64_t);
      break;
    case _NNS_UINT32:
      do_avg (uint32_t, uint64_t);
      break;
    case _NNS_FLOAT32:
      do_avg (float, long double);
      break;
    case _NNS_FLOAT64:
      do_avg (double, long double);
      break;

    default:
      assert (0);               /* Type Mismatch */
  }
  assert (inptr != outptr);

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
