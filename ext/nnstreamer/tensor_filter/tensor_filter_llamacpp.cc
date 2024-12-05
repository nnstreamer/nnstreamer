/**
 * GStreamer Tensor_Filter, llamacpp Module
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file    tensor_filter_llamacpp.cc
 * @date    5 Dec 2024
 * @brief   llamacpp module for tensor_filter gstreamer plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 *          https://github.com/ggerganov/llama.cpp
 * @author  Yelin Jeong <yelini.jeong@samsung.com>
 * @bug     No known bugs except for NYI items
 * @details Pipeline example
 *          gst-launch-1.0 filesrc location=input.txt ! application/octet-stream !
 *          tensor_converter ! other/tensors,format=flexible !
 *          tensor_filter framework=llamacpp model=llama-2-7b-chat.Q2_K.gguf invoke-dynamic=TRUE !
 *          other/tensors,format=flexible ! tensor_decoder mode=octet_stream !
 *          application/octet-stream ! filesink location=output.txt
 */

#include <assert.h>
#include <errno.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_filter_llamacpp (void) __attribute__ ((constructor));
void fini_filter_llamacpp (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

namespace nnstreamer
{
namespace tensorfilter_llamacpp
{
/**
 * @brief Class for TensorFilterLlamaCpp subplugin
 */
class TensorFilterLlamaCpp : public tensor_filter_subplugin
{
  public:
  TensorFilterLlamaCpp ();
  ~TensorFilterLlamaCpp ();

  /* mandatory methods */
  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void invoke_dynamic (GstTensorFilterProperties *prop,
      const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  /* static methods */
  static void init_filter_llama ();
  static void fini_filter_llama ();

  private:
  static TensorFilterLlamaCpp *registered;
  static const char *name;

  llama_context *ctx;
  llama_model *model;
  llama_sampler *smpl;
  llama_model_params model_params;
  llama_sampler_chain_params sampler_params;
  llama_context_params ctx_params;
};

void init_filter_llama (void) __attribute__ ((constructor));
void fini_filter_llama (void) __attribute__ ((destructor));

TensorFilterLlamaCpp *TensorFilterLlamaCpp::registered = nullptr;
const char *TensorFilterLlamaCpp::name = "llamacpp";

/**
 * @brief Construct a new llamacpp subplugin instance
 */
TensorFilterLlamaCpp::TensorFilterLlamaCpp ()
    : tensor_filter_subplugin (), ctx (nullptr), model (nullptr), smpl (nullptr)
{
}

/**
 * @brief Destructor of TensorFilterllamacpp
 */
TensorFilterLlamaCpp::~TensorFilterLlamaCpp ()
{
  if (model) {
    llama_free_model (model);
    model = nullptr;
  }

  if (smpl) {
    llama_sampler_free (smpl);
    smpl = nullptr;
  }
}

/**
 * @brief Method to get an empty object
 */
tensor_filter_subplugin &
TensorFilterLlamaCpp::getEmptyInstance ()
{
  return *(new TensorFilterLlamaCpp ());
}

/**
 * @brief Configure llamacpp instance
 */
void
TensorFilterLlamaCpp::configure_instance (const GstTensorFilterProperties *prop)
{
  /** @todo: Get ngl value from prop */
  int ngl = 99;

  if (!prop->invoke_dynamic) {
    throw std::invalid_argument (
        "llamacpp only supports invoke-dynamic mode. Set `invoke-dynamic=true`");
  }

  /** load dynamic backends */
  ggml_backend_load_all ();
  model_params = llama_model_default_params ();
  model_params.n_gpu_layers = ngl;

  model = llama_load_model_from_file ((char *) prop->model_files[0], model_params);
  if (model == nullptr) {
    throw std::invalid_argument ("Failed to load model");
  }

  ctx_params = llama_context_default_params ();

  /** enable performance counters */
  ctx_params.no_perf = false;

  sampler_params = llama_sampler_chain_default_params ();
  sampler_params.no_perf = false;
  smpl = llama_sampler_chain_init (sampler_params);

  llama_sampler_chain_add (smpl, llama_sampler_init_greedy ());
}

/**
 * @brief Invoke llamacpp using input tensors
 */
void
TensorFilterLlamaCpp::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  UNUSED (input);
  UNUSED (output);

  throw std::runtime_error (
      "llamacpp only supports invoke-dynamic mode. Set `invoke-dynamic=true`");
}

/**
 * @brief Invoke llamacpp using input tensors
 */
void
TensorFilterLlamaCpp::invoke_dynamic (GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  /** @todo: Get n_predict value from prop */
  int n_prompt, n_pos, n_predict = 32, n_decode = 0;
  llama_token new_token_id;
  llama_batch batch;
  std::string prompt, result = "";

  UNUSED (prop);

  prompt.assign ((char *) input[0].data, input[0].size);
  prompt.erase (prompt.find_last_not_of (" \t\n\r\f\v") + 1);

  n_prompt = -llama_tokenize (model, prompt.c_str (), prompt.size (), NULL, 0, true, true);

  std::vector<llama_token> prompt_tokens (n_prompt);
  if (llama_tokenize (model, prompt.c_str (), prompt.size (),
          prompt_tokens.data (), prompt_tokens.size (), true, true)
      < 0) {
    throw std::invalid_argument ("Failed to tokenize the prompt");
  }

  ctx_params.n_ctx = n_prompt + n_predict - 1;
  /** n_batch is the maximum number of tokens that can be processed in a single call to llama_decode */
  ctx_params.n_batch = n_prompt;

  ctx = llama_new_context_with_model (model, ctx_params);

  if (ctx == nullptr) {
    throw std::invalid_argument ("Failed to create llama context");
  }

  for (auto id : prompt_tokens) {
    char buf[128];
    int n = llama_token_to_piece (model, id, buf, sizeof (buf), 0, true);
    if (n < 0) {
      throw std::invalid_argument ("Failed to convert token to piece");
    }
    std::string s (buf, n);
    result += s;
  }

  batch = llama_batch_get_one (prompt_tokens.data (), prompt_tokens.size ());

  for (n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
    /** evaluate the current batch with the transformer model */
    if (llama_decode (ctx, batch)) {
      throw std::invalid_argument ("Failed to eval");
    }

    n_pos += batch.n_tokens;

    /** sample the next token */
    {
      new_token_id = llama_sampler_sample (smpl, ctx, -1);

      /** is it an end of generation? */
      if (llama_token_is_eog (model, new_token_id)) {
        break;
      }

      char buf[128];
      int n = llama_token_to_piece (model, new_token_id, buf, sizeof (buf), 0, true);
      if (n < 0) {
        throw std::invalid_argument ("Failed to convert token to piece");
      }
      std::string s (buf, n);
      result += s;

      /** prepare the next batch with the sampled token */
      batch = llama_batch_get_one (&new_token_id, 1);

      n_decode += 1;
    }
  }

  output[0].size = result.length ();
  output[0].data = strndup (result.c_str (), result.length ());
  gst_tensors_info_init (&prop->output_meta);
  prop->output_meta.num_tensors = 1;
  prop->output_meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  prop->output_meta.info[0].type = _NNS_UINT8;
  prop->output_meta.info[0].dimension[0] = output[0].size;

  llama_free (ctx);
  ctx = nullptr;
}


/**
 * @brief Get llama framework info.
 */
void
TensorFilterLlamaCpp::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 1;
  info.run_without_model = 0;
  info.verify_model_path = 1;
}

/**
 * @brief Get llama model info.
 */
int
TensorFilterLlamaCpp::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  switch (ops) {
    case GET_IN_OUT_INFO:
      in_info.num_tensors = 1;
      in_info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

      out_info.num_tensors = 1;
      out_info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
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
TensorFilterLlamaCpp::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return 0;
}

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
TensorFilterLlamaCpp::init_filter_llama ()
{
  registered = tensor_filter_subplugin::register_subplugin<TensorFilterLlamaCpp> ();
}

/** @brief Destruct the subplugin */
void
TensorFilterLlamaCpp::fini_filter_llama ()
{
  /* internal logic error */
  assert (registered != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registered);
}

} /* namespace tensorfilter_llamacpp */
} /* namespace nnstreamer */

/**
 * @brief Subplugin initializer
 */
void
init_filter_llamacpp ()
{
  nnstreamer::tensorfilter_llamacpp::TensorFilterLlamaCpp::init_filter_llama ();
}

/**
 * @brief Subplugin finalizer
 */
void
fini_filter_llamacpp ()
{
  nnstreamer::tensorfilter_llamacpp::TensorFilterLlamaCpp::fini_filter_llama ();
}
