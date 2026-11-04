/**
 * NNStreamer custom filter to test the result of invoke
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnscustom_invoke_result.c
 * @date  04 Nov 2026
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief  A pass-through custom filter whose invoke returns the number given as 'custom'.
 * @bug  No known bugs
 *
 * With 'custom=-5', every invoke fails with -5; with 'custom=1', every invoke asks
 * tensor_filter to drop the buffer. Without 'custom', it copies the input and returns 0.
 */

#include <string.h>
#include <glib.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void *
ir_init (const GstTensorFilterProperties * prop)
{
  int *result = g_new0 (int, 1);

  if (prop->custom_properties)
    *result = (int) g_ascii_strtoll (prop->custom_properties, NULL, 10);

  return result;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void
ir_exit (void *_data, const GstTensorFilterProperties * prop)
{
  UNUSED (prop);

  g_free (_data);
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
ir_setInputDim (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  UNUSED (_data);
  UNUSED (prop);

  gst_tensors_info_copy (out_info, in_info);
  return 0;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
ir_invoke (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  int result = *(int *) _data;
  unsigned int i;

  if (result != 0)
    return result;

  if (prop->input_meta.num_tensors != prop->output_meta.num_tensors)
    return -1;

  for (i = 0; i < prop->input_meta.num_tensors; i++)
    memcpy (output[i].data, input[i].data, input[i].size);

  return 0;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = ir_init,
  .exitfunc = ir_exit,
  .setInputDim = ir_setInputDim,
  .invoke = ir_invoke,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
