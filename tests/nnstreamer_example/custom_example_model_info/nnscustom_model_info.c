/**
 * NNStreamer custom filter whose model info getters can be told to fail
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnscustom_model_info.c
 * @date  02 Nov 2026
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief  A custom filter with getInputDim and getOutputDim of configurable return values.
 * @bug  No known bugs
 *
 * A test case sets 'nnscustom_model_info_input_ret' and 'nnscustom_model_info_output_ret',
 * and reads from 'nnscustom_model_info_input_calls' and 'nnscustom_model_info_output_calls'
 * which getters have run.
 */

#include <glib.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

gint nnscustom_model_info_input_ret = 0; /**< The return value of getInputDim */
gint nnscustom_model_info_output_ret = 0; /**< The return value of getOutputDim */
gint nnscustom_model_info_input_calls = 0; /**< getInputDim calls so far */
gint nnscustom_model_info_output_calls = 0; /**< getOutputDim calls so far */

/**
 * @brief Describe the single tensor of this filter.
 * @param[out] info The info to fill.
 */
static void
mi_fill_info (GstTensorsInfo * info)
{
  info->num_tensors = 1;
  info->info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:4:4:1", info->info[0].dimension);
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void *
mi_init (const GstTensorFilterProperties * prop)
{
  UNUSED (prop);

  return g_malloc0 (1);
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void
mi_exit (void *_data, const GstTensorFilterProperties * prop)
{
  UNUSED (prop);

  g_free (_data);
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
mi_getInputDim (void *_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  UNUSED (_data);
  UNUSED (prop);

  nnscustom_model_info_input_calls++;
  if (nnscustom_model_info_input_ret == 0)
    mi_fill_info (info);

  return nnscustom_model_info_input_ret;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
mi_getOutputDim (void *_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  UNUSED (_data);
  UNUSED (prop);

  nnscustom_model_info_output_calls++;
  if (nnscustom_model_info_output_ret == 0)
    mi_fill_info (info);

  return nnscustom_model_info_output_ret;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 * @details The tests query the model info only, so this always fails.
 */
static int
mi_invoke (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  UNUSED (_data);
  UNUSED (prop);
  UNUSED (input);
  UNUSED (output);

  return -1;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = mi_init,
  .exitfunc = mi_exit,
  .getInputDim = mi_getInputDim,
  .getOutputDim = mi_getOutputDim,
  .invoke = mi_invoke,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
