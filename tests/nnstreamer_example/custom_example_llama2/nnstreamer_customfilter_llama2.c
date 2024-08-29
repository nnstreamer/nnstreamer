/**
 * NNStreamer Custom Filter Example llama2.c
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file	nnstreamer_customfilter_llama2.c
 * @date	29 Aug 2024
 * @brief	Custom NNStreamer Filter Example llama2.c
 * @author	Yelin Jeong <yelini.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 * @see     https://github.com/karpathy/llama2.c
 */

#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>

/**
 * @brief init callback of tensor_filter custom
 */
static void *
llama_init (const GstTensorFilterProperties * prop)
{
  UNUSED (prop);

  return NULL;
}

/**
 * @brief exit callback of tensor_filter custom
 */
static void
llama_exit (void *private_data, const GstTensorFilterProperties * prop)
{
  UNUSED (private_data);
  UNUSED (prop);
}

/**
 * @brief setInputDimension callback of tensor_filter custom
 */
static int
get_inputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  UNUSED (private_data);
  UNUSED (prop);
  UNUSED (info);
  return 0;
}

/**
 * @brief getOutputDimension callback of tensor_filter custom
 */
static int
get_outputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  UNUSED (private_data);
  UNUSED (prop);
  UNUSED (info);
  return 0;
}

/**
 * @brief invoke callback of tensor_filter custom
 */
static int
llama_invoke (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  UNUSED (private_data);
  UNUSED (prop);
  UNUSED (input);
  UNUSED (output);

  return 0;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = llama_init,
  .exitfunc = llama_exit,
  .getInputDim = get_inputDim,
  .getOutputDim = get_outputDim,
  .invoke = llama_invoke,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
