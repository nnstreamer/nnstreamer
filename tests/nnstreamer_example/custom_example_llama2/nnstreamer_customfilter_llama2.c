/**
 * NNStreamer Custom Filter Example llama2.c
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file	nnstreamer_customfilter_llama2.c
 * @date	06 Sep 2024
 * @brief	Custom NNStreamer Filter Example llama2.c
 * @author	Yelin Jeong <yelini.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 * @see     https://github.com/nnsuite/llama2.c
 * @todo    Use property to values like temperatrue, topp
 */


#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>
#include <tensor_filter_custom.h>
#include <assert.h>
#include "api.h"

#define CHECKPOINT_PATH "model.bin"
#define TOKENIZER_PATH "tokenizer.bin"

#define TEMPERATURE \
  1.0f // 0.0 = greedy deterministic. 1.0 = original. don't set higher
#define TOPP \
  0.9f // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
#define RNG_SEED 0ULL // seed rng with time by default

/**
 * @brief _pt_data Internal data structure
 */
typedef struct _pt_data {
  Transformer *transformer;
  Tokenizer *tokenizer;
  Sampler *sampler;
  char *prompt; // prompt string
  int steps; // number of steps to run for

  GstTensorsInfo info;
} pt_data;

/**
 * @brief init callback of tensor_filter custom
 */
static void *
pt_init (const GstTensorFilterProperties *prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));
  Config *config;
  UNUSED (prop);

  gst_tensors_info_init (&data->info);

  data->transformer = (Transformer *) malloc (sizeof (Transformer));

  // build the Transformer via the model .bin file
  build_transformer (data->transformer, (char *) CHECKPOINT_PATH);
  config = &data->transformer->config;

  if (data->steps == 0 || data->steps > config->seq_len)
    data->steps = config->seq_len; // override to ~max length

  data->tokenizer = (Tokenizer *) malloc (sizeof (Tokenizer));

  // build the Tokenizer via the tokenizer .bin file
  build_tokenizer (data->tokenizer, (char *) TOKENIZER_PATH, config->vocab_size);

  data->sampler = (Sampler *) malloc (sizeof (Sampler));

  // build the Sampler
  build_sampler (data->sampler, config->vocab_size, TEMPERATURE, TOPP, RNG_SEED);

  data->info.num_tensors = 1;
  data->info.info[0].type = _NNS_UINT8;
  data->info.info[0].dimension[0] = data->tokenizer->max_token_length * data->steps;

  return data;
}

/**
 * @brief exit callback of tensor_filter custom
 */
static void
pt_exit (void *private_data, const GstTensorFilterProperties *prop)
{
  pt_data *data = private_data;
  UNUSED (prop);
  assert (private_data);

  gst_tensors_info_free (&data->info);
  free_transformer (data->transformer);
  free_tokenizer (data->tokenizer);
  free_sampler (data->sampler);

  free (data->transformer);
  free (data->tokenizer);
  free (data->sampler);
  free (data);
}

/**
 * @brief setInputDimension callback of tensor_filter custom
 */
static int
get_inputDim (void *private_data, const GstTensorFilterProperties *prop, GstTensorsInfo *info)
{
  UNUSED (private_data);
  UNUSED (prop);
  assert (info);

  info->format = _NNS_TENSOR_FORMAT_SPARSE;
  info->num_tensors = 1;

  return 0;
}

/**
 * @brief getOutputDimension callback of tensor_filter custom
 */
static int
get_outputDim (void *private_data, const GstTensorFilterProperties *prop, GstTensorsInfo *info)
{
  pt_data *data = private_data;
  UNUSED (prop);
  assert (private_data);

  gst_tensors_info_copy (info, &data->info);
  return 0;
}

/**
 * @brief invoke callback of tensor_filter custom
 */
static int
pt_invoke (void *private_data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  pt_data *data = private_data;
  char *result, *prompt;
  int len;

  UNUSED (prop);
  assert (private_data);
  assert (input);
  assert (output);

  prompt = (char *) input[0].data;
  len = strlen (prompt);

  if (len > 1) {
    prompt[strlen (prompt) - 1] = ' '; // Remove new line from text file
  } else {
    prompt = NULL;
  }

  output->size = data->tokenizer->max_token_length * data->steps;

  result = get_tokens (data->transformer, data->tokenizer, data->sampler, prompt, data->steps);
  memcpy (output[0].data, result, strlen(result));

  free (result);
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
