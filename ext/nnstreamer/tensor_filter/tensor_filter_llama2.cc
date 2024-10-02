/**
 * GStreamer Tensor_Filter, llama2c Module
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file    tensor_filter_llama2.cc
 * @date    19 Sep 2024
 * @brief   llama2c module for tensor_filter gstreamer plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 *          https://github.com/nnsuite/llama2.c
 * @author  Yelin Jeong <yelini.jeong@samsung.com>
 * @bug     No known bugs except for NYI items
 * @todo    Use property to values like temperatrue, topp
 * @details Pipeline example
 *          gst-launch-1.0 filesrc location=input.txt ! application/octet-stream !
 *          tensor_converter ! other/tensors,format=flexible !
 *          tensor_filter framework=llama2c model=model.bin,tokenizer.bin !
 *          filesink location=output.txt
 */

#include <assert.h>
#include <errno.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_util.h>

#define TEMPERATURE \
  1.0f // 0.0 = greedy deterministic. 1.0 = original. don't set higher
#define TOPP \
  0.9f // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
#define RNG_SEED 0ULL // seed rng with time by default

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_filter_llama2c (void) __attribute__ ((constructor));
void fini_filter_llama2c (void) __attribute__ ((destructor));
#include "api.h"
#ifdef __cplusplus
}
#endif /* __cplusplus */

namespace nnstreamer
{
namespace tensorfilter_llama2c
{
/**
 * @brief Class for llama2 subplugin
 */
class TensorFilterLlama2c : public tensor_filter_subplugin
{
  public:
  TensorFilterLlama2c ();
  ~TensorFilterLlama2c ();

  /* mandatory methods */
  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  /* static methods */
  static void init_filter_llama2c ();
  static void fini_filter_llama2c ();

  private:
  static TensorFilterLlama2c *registered;
  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw;

  static Transformer *transformer;
  static Tokenizer *tokenizer;
  static Sampler *sampler;
  static Config *config;
  char *prompt; // prompt string
  int steps; // number of steps to run for

  static void *allocate (size_t size);
};

void init_filter_llama2c (void) __attribute__ ((constructor));
void fini_filter_llama2c (void) __attribute__ ((destructor));

TensorFilterLlama2c *TensorFilterLlama2c::registered = nullptr;
const char *TensorFilterLlama2c::name = "llama2c";
Transformer *TensorFilterLlama2c::transformer = nullptr;
Tokenizer *TensorFilterLlama2c::tokenizer = nullptr;
Sampler *TensorFilterLlama2c::sampler = nullptr;
Config *TensorFilterLlama2c::config = nullptr;

/**
 * @brief Construct a new llama2c subplugin instance
 */
TensorFilterLlama2c::TensorFilterLlama2c ()
{
}

/**
 * @brief Destructor of TensorFilterLlama2c
 */
TensorFilterLlama2c::~TensorFilterLlama2c ()
{
}

/**
 * @brief Method to get an empty object
 */
tensor_filter_subplugin &
TensorFilterLlama2c::getEmptyInstance ()
{
  return *(new TensorFilterLlama2c ());
}

/**
 * @brief Configure llama2c instance
 */
void
TensorFilterLlama2c::configure_instance (const GstTensorFilterProperties *prop)
{
  // build the Transformer via the model file
  build_transformer (transformer, (char *) prop->model_files[0]);
  config = &transformer->config;

  if (steps == 0 || steps > config->seq_len)
    steps = config->seq_len; // override to ~max length

  // build the Tokenizer via the tokenizer file
  build_tokenizer (tokenizer, (char *) prop->model_files[1], config->vocab_size);

  // build the Sampler
  build_sampler (sampler, config->vocab_size, TEMPERATURE, TOPP, RNG_SEED);
}

/**
 * @brief Invoke llama2c using input tensors
 */
void
TensorFilterLlama2c::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  char *result, *prompt;
  size_t len;

  prompt = (char *) allocate (input[0].size + 1);
  memcpy (prompt, input[0].data, input[0].size);
  prompt[input[0].size] = '\0';

  len = strlen (prompt);

  if (len > 0) {
    /* Remove new line from text file */
    char c = prompt[len - 1];
    if (c == '\r' || c == '\n')
      prompt[len - 1] = '\0';
  }

  result = get_tokens (transformer, tokenizer, sampler, prompt, steps);

  /** @todo Make out_info flexible */
  output[0].size = tokenizer->max_token_length * steps;
  output[0].data = allocate (output[0].size);

  if (result) {
    len = strlen (result);
    if (len > 0)
      memcpy (output[0].data, result, len);
    free (result);
  }

  free (prompt);
}

/**
 * @brief Get llama2c framework info.
 */
void
TensorFilterLlama2c::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 1;
  info.run_without_model = 0;
  info.verify_model_path = 1;
}

/**
 * @brief Get llama2c model info.
 */
int
TensorFilterLlama2c::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  switch (ops) {
    case GET_IN_OUT_INFO:
      in_info.num_tensors = 1;
      in_info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

      /** @todo Make out_info flexible */
      out_info.num_tensors = 1;
      out_info.format = _NNS_TENSOR_FORMAT_STATIC;
      out_info.info[0].type = _NNS_UINT8;
      out_info.info[0].dimension[0] = tokenizer->max_token_length * steps;
      break;
    default:
      return -ENOENT;
  }

  return 0;
}

/**
 * @brief Method to handle the event
 */
int
TensorFilterLlama2c::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return 0;
}

/**
 * @brief Allocate new memory.
 */
void *
TensorFilterLlama2c::allocate (size_t size)
{
  void *mem = nullptr;

  if (size > 0)
    mem = malloc (size);

  assert (mem != nullptr);
  memset (mem, 0, size);

  return mem;
}

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
TensorFilterLlama2c::init_filter_llama2c ()
{
  transformer = (Transformer *) allocate (sizeof (Transformer));
  tokenizer = (Tokenizer *) allocate (sizeof (Tokenizer));
  sampler = (Sampler *) allocate (sizeof (Sampler));
  registered = tensor_filter_subplugin::register_subplugin<TensorFilterLlama2c> ();
}

/** @brief Destruct the subplugin */
void
TensorFilterLlama2c::fini_filter_llama2c ()
{
  free_transformer (transformer);
  free_tokenizer (tokenizer);
  free_sampler (sampler);

  free (transformer);
  free (tokenizer);
  free (sampler);

  transformer = nullptr;
  tokenizer = nullptr;
  sampler = nullptr;
  config = nullptr;

  /* internal logic error */
  assert (registered != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registered);
}

} /* namespace tensorfilter_llama2c */
} /* namespace nnstreamer */

/**
 * @brief Subplugin initializer
 */
void
init_filter_llama2c ()
{
  nnstreamer::tensorfilter_llama2c::TensorFilterLlama2c::init_filter_llama2c ();
}

/**
 * @brief Subplugin finalizer
 */
void
fini_filter_llama2c ()
{
  nnstreamer::tensorfilter_llama2c::TensorFilterLlama2c::fini_filter_llama2c ();
}
