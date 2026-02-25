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
 *          tensor_filter framework=llamacpp model=llama-2-7b-chat.Q2_K.gguf invoke-dynamic=TRUE
 *          custom=num_predict:32 ! other/tensors,format=flexible ! tensor_decoder mode=octet_stream !
 *          application/octet-stream ! filesink location=output.txt
 */

#include <assert.h>
#include <atomic>
#include <condition_variable>
#include <errno.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <queue>
#include <string>
#include <thread>
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

  static constexpr float TOP_P_DISABLED
      = 1.1f; /* Top-P sampling disabled (valid range: 0.0 to 1.0) */
  static constexpr float TYPICAL_P_DISABLED
      = 1.1f; /* Typical-P sampling disabled (valid range: 0.0 to 1.0) */
  static constexpr float TEMPERATURE_DISABLED
      = -1.0f; /* Disable temperature sampling (valid range: 0.0 or greater) */

  llama_model *model;
  llama_context *ctx;
  llama_sampler *sampler;
  llama_adapter_lora *lora_adapter;

  int n_processed = 0;
  int n_gpu_layers = 0;
  int n_predict = 32;
  int n_ctx = 0;
  int n_batch = 512;
  int top_k = 0;
  float top_p = TOP_P_DISABLED;
  float typical_p = TYPICAL_P_DISABLED;
  float temperature = TEMPERATURE_DISABLED;
  std::string save_ctx_path;
  std::string load_ctx_path;

  std::thread output_thread; /* thread for async output */
  std::queue<std::pair<GstTensorFilterProperties *, std::string>> input_queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::atomic<bool> stop_thread{ false }; /* flag to stop the thread */
  void parseCustomProperties (const GstTensorFilterProperties *prop);
  void generateTokens (GstTensorFilterProperties *prop, GstTensorMemory *output,
      const std::string &prompt);
  bool createOutputTensor (GstTensorFilterProperties *prop,
      GstTensorMemory *output, const std::string &buf);
  bool updateOutput (GstTensorFilterProperties *prop, llama_token &token,
      std::string &accumulated);
  void outputThreadLoop ();
  void cleanupResources ();
  bool loadContextFromFile (const std::string &path);
  bool saveContextToFile (const std::string &path);

  using PropertySetter = std::function<void (const gchar *)>;
  std::map<std::string, PropertySetter> property_setters_;
  void initializePropertySetters ();
};

TensorFilterLlamaCpp *TensorFilterLlamaCpp::registered = nullptr;
const char *TensorFilterLlamaCpp::name = "llamacpp";

/**
 * @brief Construct a new llamacpp subplugin instance
 */
TensorFilterLlamaCpp::TensorFilterLlamaCpp ()
    : tensor_filter_subplugin (), model (nullptr), ctx (nullptr),
      sampler (nullptr), lora_adapter (nullptr)
{
  initializePropertySetters ();
}

/**
 * @brief Destructor of TensorFilterllamacpp
 */
TensorFilterLlamaCpp::~TensorFilterLlamaCpp ()
{
  if (!save_ctx_path.empty () && ctx) {
    if (!saveContextToFile (save_ctx_path)) {
      ml_loge ("Failed to save context to %s", save_ctx_path.c_str ());
    }
  }

  cleanupResources ();
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
 * @brief Initialize the map of property setters with lambda functions.
 */
void
TensorFilterLlamaCpp::initializePropertySetters ()
{
  property_setters_["num_predict"] = [this] (const gchar *value) {
    this->n_predict = static_cast<int> (g_ascii_strtoll (value, NULL, 10));
  };
  property_setters_["num_gpu_layers"] = [this] (const gchar *value) {
    this->n_gpu_layers = static_cast<int> (g_ascii_strtoll (value, NULL, 10));
  };
  property_setters_["context_length"] = [this] (const gchar *value) {
    this->n_ctx = static_cast<int> (g_ascii_strtoll (value, NULL, 10));
  };
  property_setters_["batch_size"] = [this] (const gchar *value) {
    this->n_batch = static_cast<int> (g_ascii_strtoll (value, NULL, 10));
  };
  property_setters_["top_k"] = [this] (const gchar *value) {
    this->top_k = static_cast<int> (g_ascii_strtoll (value, NULL, 10));
  };
  property_setters_["top_p"] = [this] (const gchar *value) {
    this->top_p = static_cast<float> (g_ascii_strtod (value, NULL));
  };
  property_setters_["typical_p"] = [this] (const gchar *value) {
    this->typical_p = static_cast<float> (g_ascii_strtod (value, NULL));
  };
  property_setters_["temperature"] = [this] (const gchar *value) {
    this->temperature = static_cast<float> (g_ascii_strtod (value, NULL));
  };
  property_setters_["save_ctx"] = [this] (const gchar *value) {
    this->save_ctx_path = std::string (value);
  };
  property_setters_["load_ctx"] = [this] (const gchar *value) {
    this->load_ctx_path = std::string (value);
  };
}

/**
 * @brief Parse custom prop and set instance options accordingly.
 */
void
TensorFilterLlamaCpp::parseCustomProperties (const GstTensorFilterProperties *prop)
{
  using uniq_g_strv = std::unique_ptr<gchar *, std::function<void (gchar **)>>;
  guint len, i;

  if (!prop->custom_properties)
    return;

  uniq_g_strv options (g_strsplit (prop->custom_properties, ",", -1), g_strfreev);
  len = g_strv_length (options.get ());

  for (i = 0; i < len; ++i) {
    uniq_g_strv option (g_strsplit (options.get ()[i], ":", -1), g_strfreev);

    if (g_strv_length (option.get ()) > 1) {
      gchar *key = g_strstrip (option.get ()[0]);
      gchar *value = g_strstrip (option.get ()[1]);

      std::string prop_name (key);
      std::transform (prop_name.begin (), prop_name.end (), prop_name.begin (),
          [] (unsigned char c) { return std::tolower (c); });
      auto it = property_setters_.find (prop_name);

      if (it != property_setters_.end ()) {
        it->second (value);
      } else {
        throw std::invalid_argument ("Unknown custom property " + prop_name);
      }
    }
  }

  /* validate properties */
  if (n_ctx < 0) {
    throw std::invalid_argument ("Context length should be a positive value (or 0 from model).");
  }
  if (n_batch <= 0) {
    throw std::invalid_argument ("Batch size should be a positive value.");
  }
}

/**
 * @brief Configure llamacpp instance
 */
void
TensorFilterLlamaCpp::configure_instance (const GstTensorFilterProperties *prop)
{
  if (!prop->invoke_dynamic) {
    throw std::invalid_argument (
        "llamacpp only supports invoke-dynamic mode. Set `invoke-dynamic=true`");
  }

  try {
    parseCustomProperties (prop);
  } catch (const std::invalid_argument &e) {
    throw std::invalid_argument ("Failed to parse \"custom\" prop:"
                                 + std::string (e.what ()) + "\n\tReference: " + __FILE__);
  }

  /* load dynamic backends */
  ggml_backend_load_all ();

  /* initialize model */
  llama_model_params model_params = llama_model_default_params ();
  model_params.n_gpu_layers = n_gpu_layers;

  model = llama_model_load_from_file (prop->model_files[0], model_params);
  if (model == nullptr) {
    cleanupResources ();
    throw std::runtime_error ("Failed to load model");
  }

  llama_context_params ctx_params = llama_context_default_params ();
  ctx_params.n_ctx = n_ctx;
  /* n_batch is the maximum number of tokens that can be processed in a single call to llama_decode */
  ctx_params.n_batch = n_batch;
  /* enable performance counters */
  ctx_params.no_perf = false;

  ctx = llama_init_from_model (model, ctx_params);
  if (ctx == nullptr) {
    cleanupResources ();
    throw std::runtime_error ("Failed to create llama context");
  }

  /* Check for LoRA adapter in model_files */
  if (prop->model_files[1] != nullptr) {
    lora_adapter = llama_adapter_lora_init (model, prop->model_files[1]);
    if (lora_adapter == nullptr) {
      cleanupResources ();
      throw std::runtime_error ("Failed to load LoRA adapter");
    }
    llama_set_adapter_lora (ctx, lora_adapter, 1.0f);
    ml_logd ("Successfully applied LoRA adapter to the context");
  }

  if (!load_ctx_path.empty ()) {
    if (std::filesystem::exists (load_ctx_path)) {
      if (!loadContextFromFile (load_ctx_path)) {
        ml_logw ("Failed to load context from %s, continuing with fresh context",
            load_ctx_path.c_str ());
      }
    } else {
      ml_logw ("Context file %s not found, starting with fresh context",
          load_ctx_path.c_str ());
    }
  }

  /* initialize sampler */
  llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params ();
  sampler_params.no_perf = false;

  sampler = llama_sampler_chain_init (sampler_params);
  if (sampler == nullptr) {
    cleanupResources ();
    throw std::runtime_error ("Failed to initialize sampler");
  }

  if (top_k > 0) {
    llama_sampler_chain_add (sampler, llama_sampler_init_top_k (top_k));
  }
  if (top_p > 0.0f && top_p < 1.0f) {
    llama_sampler_chain_add (sampler, llama_sampler_init_top_p (top_p, 1));
  }
  if (typical_p > 0.0f && typical_p < 1.0f) {
    llama_sampler_chain_add (sampler, llama_sampler_init_typical (typical_p, 1));
  }
  if (temperature >= 0.0f) {
    llama_sampler_chain_add (sampler, llama_sampler_init_temp (temperature));
  }
  /* if we use 'llama_sampler_init_greedy()' : Roulette based on probability */
  llama_sampler_chain_add (sampler, llama_sampler_init_greedy ());

  stop_thread = false;
  output_thread = std::thread (&TensorFilterLlamaCpp::outputThreadLoop, this);
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
 * @brief Create output tensor
 *
 * The memory allocated for the output tensor data using strndup
 * must be freed by the caller of this function to avoid memory leaks.
 */
bool
TensorFilterLlamaCpp::createOutputTensor (GstTensorFilterProperties *prop,
    GstTensorMemory *output, const std::string &buf)
{
  GstTensorInfo *_info;

  if (buf.empty () || output == nullptr || prop == nullptr) {
    ml_loge ("Invalid arguments passed to createOutputTensor");
    return false;
  }

  output[0].size = buf.length (); /* Stores string length, excluding NULL appended to the end of the string */
  output[0].data = strndup (buf.c_str (), buf.length ());

  if (output[0].data == nullptr) {
    ml_loge ("createOutputTensor: strndup failed");
    return false;
  }

  gst_tensors_info_free (&prop->output_meta);

  prop->output_meta.num_tensors = 1;
  prop->output_meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  _info = gst_tensors_info_get_nth_info (&prop->output_meta, 0);
  _info->type = _NNS_UINT8;
  _info->dimension[0] = output[0].size;

  return true;
}

/**
 * @brief Update output string.
 */
bool
TensorFilterLlamaCpp::updateOutput (GstTensorFilterProperties *prop,
    llama_token &token, std::string &accumulated)
{
  const llama_vocab *vocab = llama_model_get_vocab (model);
  char buf[128] = { 0 };
  int n = llama_token_to_piece (vocab, token, buf, sizeof (buf), 0, true);

  if (n < 0) {
    ml_loge ("Failed to convert token to piece.");
    return false;
  }

  /* Dispatch output data when invoke-async is enabled. */
  if (prop->invoke_async) {
    GstTensorMemory output;

    if (!createOutputTensor (prop, &output, std::string (buf, n))) {
      ml_loge ("Failed to create output tensor.");
      return false;
    }

    nnstreamer_filter_dispatch_output_async (prop, &output);
  } else {
    accumulated.append (buf, n);
  }

  return true;
}

/**
 * @brief Generates output tokens from the given input prompt using llamacpp.
 *
 * If invoke_async is true, each token is dispatched asynchronously using
 * nnstreamer_filter_dispatch_output_async(); otherwise, the result is stored in
 * output. GstTensorMemory is freed by nnstreamer; do not free it manually.
 */
void
TensorFilterLlamaCpp::generateTokens (GstTensorFilterProperties *prop,
    GstTensorMemory *output, const std::string &prompt)
{
  const llama_vocab *vocab = llama_model_get_vocab (model);
  int n;
  llama_token new_token_id;
  llama_batch batch;
  std::string output_accumulated;

  std::vector<llama_token> prompt_tokens (prompt.length ());
  n = llama_tokenize (vocab, prompt.c_str (), prompt.size (),
      prompt_tokens.data (), prompt_tokens.size (), false, true);
  if (n < 0) {
    throw std::runtime_error ("Failed to tokenize the prompt");
  }
  prompt_tokens.resize (n);

  if (n_processed + (int) prompt_tokens.size () > n_ctx) {
    const int n_keep = n_ctx / 2;
    ml_logd ("Context is full, trimming tokens to %d.", n_keep);
    llama_memory_seq_rm (llama_get_memory (ctx), 0, n_keep, n_processed);
    llama_memory_seq_add (
        llama_get_memory (ctx), 0, n_keep, n_processed, -(n_processed - n_keep));
    n_processed = n_keep;
  }

  batch = llama_batch_get_one (prompt_tokens.data (), prompt_tokens.size ());
  /* Since batch.pos is NULL, context will use internal n_processed state */
  if (llama_decode (ctx, batch)) {
    throw std::runtime_error ("Failed to eval");
  }
  n_processed += prompt_tokens.size ();

  int n_remain = n_predict;
  while (n_remain > 0) {
    /* evaluate the current batch with the transformer model */
    /* sample the next token */
    new_token_id = llama_sampler_sample (sampler, ctx, -1);
    llama_sampler_accept (sampler, new_token_id);

    /* is it an end of generation? */
    if (llama_vocab_is_eog (vocab, new_token_id)) {
      break;
    }

    if (!updateOutput (prop, new_token_id, output_accumulated)) {
      throw std::runtime_error ("Failed to update output from generated token");
    }
    n_remain--;

    batch = llama_batch_get_one (&new_token_id, 1);
    if (llama_decode (ctx, batch)) {
      throw std::runtime_error ("Failed to eval token");
    }
    n_processed++;
  }

  if (!prop->invoke_async) {
    /* Final output for synchronous mode is created here. */
    if (!createOutputTensor (prop, output, output_accumulated)) {
      throw std::runtime_error ("Failed to create output tensor while generating tokens");
    }
  }
}

/**
 * @brief Asynchronous output thread loop. This thread generates tokens
 */
void
TensorFilterLlamaCpp::outputThreadLoop ()
{
  while (!stop_thread) {
    std::unique_lock<std::mutex> lock (queue_mutex);

    queue_cv.wait (lock, [this] () {
      return !input_queue.empty () || stop_thread;
    }); /*wait when condition is false*/

    if (stop_thread)
      break;

    auto [prop, prompt] = std::move (input_queue.front ());
    input_queue.pop ();
    lock.unlock ();
    try {
      this->generateTokens (prop, NULL, prompt);
    } catch (const std::exception &e) {
      /* thread exception */
      ml_loge ("Exception occurred during token generation: %s", e.what ());
    }
  }
}

/**
 * @brief Clean up allocated resources
 */
void
TensorFilterLlamaCpp::cleanupResources ()
{
  stop_thread = true;
  queue_cv.notify_one ();
  if (output_thread.joinable ()) {
    output_thread.join ();
  }

  {
    std::lock_guard<std::mutex> lock (queue_mutex);
    while (!input_queue.empty ())
      input_queue.pop ();
  }

  if (sampler) {
    llama_sampler_free (sampler);
    sampler = nullptr;
  }

  if (lora_adapter) {
    llama_adapter_lora_free (lora_adapter);
    lora_adapter = nullptr;
  }

  if (ctx) {
    llama_free (ctx);
    ctx = nullptr;
  }

  if (model) {
    llama_model_free (model);
    model = nullptr;
  }
}

/**
 * @brief Load context from file
 */
bool
TensorFilterLlamaCpp::loadContextFromFile (const std::string &path)
{
  if (!ctx) {
    ml_loge ("Context not initialized for loading state");
    return false;
  }

  std::vector<uint8_t> ctx_data;
  size_t ctx_size = 0;
  std::ifstream file (path, std::ios::binary);
  try {
    file.exceptions (std::ifstream::failbit | std::ifstream::badbit);

    file.seekg (0, std::ios::end);
    ctx_size = file.tellg ();
    file.seekg (0, std::ios::beg);

    file.read (reinterpret_cast<char *> (ctx_data.data ()), ctx_size);
    file.close ();

  } catch (const std::exception &e) {
    ml_loge ("Failed to read context file: %s (file: %s)", e.what (), path.c_str ());
    file.close ();

    return false;
  }

  if (llama_state_set_data (ctx, ctx_data.data (), ctx_size) != ctx_size) {
    ml_loge ("Failed to set context data");
    return false;
  }

  ml_logi ("Successfully loaded context from %s", path.c_str ());

  return true;
}

/**
 * @brief Save context to file
 */
bool
TensorFilterLlamaCpp::saveContextToFile (const std::string &path)
{
  if (!ctx) {
    ml_loge ("Context not initialized for saving state");
    return false;
  }

  size_t ctx_size = llama_state_get_size (ctx);
  if (ctx_size == 0) {
    ml_loge ("Invalid state size");
    return false;
  }

  std::vector<uint8_t> ctx_data (ctx_size);
  size_t written_size = llama_state_get_data (ctx, ctx_data.data (), ctx_size);
  if (written_size != ctx_size) {
    ml_loge ("Failed to get context data");
    return false;
  }

  std::ofstream file (path, std::ios::binary);
  try {
    if (!file.is_open ()) {
      ml_loge ("Cannot create context file %s", path.c_str ());
      file.close ();

      return false;
    }

    file.write (reinterpret_cast<const char *> (ctx_data.data ()), ctx_size);
    file.close ();

  } catch (const std::exception &e) {
    ml_loge ("Failed to write context file: %s (file: %s)", e.what (), path.c_str ());
    file.close ();

    return false;
  }

  ml_logi ("Successfully saved context to %s", path.c_str ());
  return true;
}

/**
 * @brief Invoke llamacpp using input tensors
 */
void
TensorFilterLlamaCpp::invoke_dynamic (GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  std::string prompt;

  if (!input || !input[0].data || input[0].size == 0) {
    throw std::invalid_argument ("Invalid input tensor data");
  }

  /* The part that fills in the prompt value was moved out of the thread, because input is freed when the invoke_dynamic function returns.*/
  prompt.assign ((char *) input[0].data, input[0].size);
  prompt.erase (prompt.find_last_not_of (" \t\n\r\f\v") + 1);

  if (prop->invoke_async) {
    {
      std::lock_guard<std::mutex> lock (queue_mutex);
      input_queue.push (std::make_pair (prop, std::move (prompt)));
    }
    queue_cv.notify_one (); /* notify waiting thread */
  } else {
    generateTokens (prop, output, prompt);
  }
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
  return -ENOENT;
}

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
TensorFilterLlamaCpp::init_filter_llama ()
{
  registered = tensor_filter_subplugin::register_subplugin<TensorFilterLlamaCpp> ();
  nnstreamer_filter_set_custom_property_desc (name, "num_predict", "Number of tokens to predict",
      "num_gpu_layers", "Number of layers to offload to the GPU", "context_length",
      "Context size for KV cache", "batch_size", "Logical maximum batch size",
      "top_k", "Top-K sampling parameter", "top_p", "Top-P sampling parameter",
      "typical_p", "Typical-P sampling parameter", "temperature",
      "Temperature sampling parameter", "save_ctx", "Path to save llama context state",
      "load_ctx", "Path to load llama context state", NULL);
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
