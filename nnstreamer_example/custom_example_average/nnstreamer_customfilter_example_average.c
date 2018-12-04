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
 * @bug		No known bugs except for NYI items
 */

#include <stdlib.h>
#include <assert.h>
#include <tensor_filter_custom.h>

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
  int i;

  assert (private_data);
  assert (in_info);
  assert (out_info);

  out_info->num_tensors = 1;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    out_info->info[0].dimension[i] = in_info->info[0].dimension[i];

  /* Update output dimension [1] and [2] with new-x, new-y */
  out_info->info[0].dimension[1] = 1;
  out_info->info[0].dimension[2] = 1;

  out_info->info[0].type = in_info->info[0].type;
  return 0;
}


/**
 * @brief do_avg
 */
#define do_avg(type,sumtype) do {\
      sumtype *avg = (sumtype *) malloc(sizeof(sumtype) * prop->input_meta.info[0].dimension[0]); \
      type *iptr = (type *) input[0].data; \
      type *optr = (type *) output[0].data; \
      for (z = 0; z < prop->input_meta.info[0].dimension[3]; z++) { \
        for (y = 0; y < prop->input_meta.info[0].dimension[0]; y++) \
          avg[y] = 0; \
        for (y = 0; y < prop->input_meta.info[0].dimension[2]; y++) { \
          for (x = 0; x < prop->input_meta.info[0].dimension[1]; x++) { \
            for (c = 0; c < prop->input_meta.info[0].dimension[0]; c++) { \
              avg[c] += *(iptr + c + x * ix + y * iy + z * iz); \
            } \
          } \
        } \
        for (c = 0; c < prop->input_meta.info[0].dimension[0]; c++) { \
          *(optr + c + z * prop->input_meta.info[0].dimension[0]) = (type) (avg[c] / xy); \
        } \
      } \
      free(avg); \
  } while (0)

/**
 * @brief pt_invoke
 */
static int
pt_invoke (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  uint32_t c, x, y, z;

  uint32_t ix = prop->input_meta.info[0].dimension[0];
  uint32_t iy =
      prop->input_meta.info[0].dimension[0] *
      prop->input_meta.info[0].dimension[1];
  uint32_t iz =
      prop->input_meta.info[0].dimension[0] *
      prop->input_meta.info[0].dimension[1] *
      prop->input_meta.info[0].dimension[2];
  uint32_t xy =
      prop->input_meta.info[0].dimension[1] *
      prop->input_meta.info[0].dimension[2];

  assert (private_data);
  assert (input);
  assert (output);

  /* This assumes the limit is 4 */
  assert (NNS_TENSOR_RANK_LIMIT == 4);

  assert (prop->input_meta.info[0].dimension[0] ==
      prop->output_meta.info[0].dimension[0]);
  assert (prop->input_meta.info[0].dimension[3] ==
      prop->output_meta.info[0].dimension[3]);
  assert (prop->input_meta.info[0].type == prop->output_meta.info[0].type);

  switch (prop->input_meta.info[0].type) {
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
    case _NNS_INT64:
      do_avg (int64_t, int64_t);
      break;
    case _NNS_UINT64:
      do_avg (uint64_t, uint64_t);
      break;
    default:
      assert (0);               /* Type Mismatch */
  }
  assert (input[0].data != output[0].data);

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
