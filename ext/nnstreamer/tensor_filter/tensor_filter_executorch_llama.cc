/* SPDX-License-Identifier: LGPL-2.1-only */

/**
 * @file    tensor_filter_executorch_llama.cc
 * @date    16 Oct 2024
 * @brief   NNStreamer tensor-filter sub-plugin for ExecuTorch Llama
 * @author
 * @see     https://github.com/nnstreamer/nnstreamer
 *          https://github.com/pytorch/executorch/tree/main/extension/llm/runner
 * @bug     No known bugs.
 * @note    Currently experimental.
 * @details A pipeline example:
 *          gst-launch-1.0  filesrc location=input.txt ! application/octet-stream !
 *          tensor_converter ! other/tensors,format=flexible !
 *          tensor_filter framework=executorch-llama model=model.pte,tokenizer.model invoke-dynamic=true !
 *          other/tensors,format=flexible ! tensor_decoder mode=octet_stream !
 *          application/octet-stream ! filesink location=result.txt
 */

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <glib.h>
#include <mutex>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <queue>
#include <thread>
#include <unordered_set>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/module/module.h>

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

namespace nnstreamer
{
namespace tensorfilter_executorch_llama
{

#ifdef __cplusplus
extern "C" {
#endif
void init_filter_executorch_llama (void) __attribute__ ((constructor));
void fini_filter_executorch_llama (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief tensor-filter-subplugin concrete class for ExecuTorch-Llama
 */
class executorch_llama_subplugin final : public tensor_filter_subplugin
{
  private:
  static const char *fw_name;
  static executorch_llama_subplugin *registeredRepresentation;

  /* executorch */
  std::unique_ptr<llm::TextLLMRunner> runner; /**< Runner for Llama model */
  std::unordered_set<std::string> eos_texts; /**< Text of the tokens that end generation */
  llm::GenerationConfig config; /**< Generation options of the custom property */
  bool keep_eos = false; /**< Keep the text of the end-of-sequence tokens */

  /* invoke-async */
  std::thread worker; /**< Generates the queued prompts */
  std::mutex queue_lock; /**< Guards prompts */
  std::condition_variable queue_cond; /**< Signals a queued prompt or stopping */
  std::queue<std::pair<GstTensorFilterProperties *, std::string>> prompts; /**< Prompts waiting for the worker */
  std::atomic<bool> stopping{ false }; /**< The instance is being closed */

  void parseCustomProperties (const char *custom);
  std::string generate (GstTensorFilterProperties *prop, const std::string &prompt);
  void dispatchPiece (GstTensorFilterProperties *prop, const std::string &piece);
  void workerLoop ();

  public:
  static void init_filter_executorch_llama ();
  static void fini_filter_executorch_llama ();

  /** @brief Construct an unconfigured subplugin instance */
  executorch_llama_subplugin () = default;
  ~executorch_llama_subplugin ();

  tensor_filter_subplugin &getEmptyInstance () override;
  void configure_instance (const GstTensorFilterProperties *prop) override;
  void invoke (const GstTensorMemory *input, GstTensorMemory *output) override;
  void invoke_dynamic (GstTensorFilterProperties *prop,
      const GstTensorMemory *input, GstTensorMemory *output) override;
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info) override;
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info,
      GstTensorsInfo &out_info) override;
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data) override;
};

const char *executorch_llama_subplugin::fw_name = "executorch-llama";

/**
 * @brief Destruct the instance, releasing the Llama runner. The generation in
 *        progress is stopped and the prompts still queued are dropped.
 */
executorch_llama_subplugin::~executorch_llama_subplugin ()
{
  {
    std::lock_guard<std::mutex> lock (queue_lock);
    stopping = true;
  }
  queue_cond.notify_one ();
  if (worker.joinable ())
    worker.join ();
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
executorch_llama_subplugin::getEmptyInstance ()
{
  return *(new executorch_llama_subplugin ());
}

/**
 * @brief Parse an integer option value within [min, max].
 */
static int32_t
parse_int_option (const gchar *key, const gchar *value, gint64 min, gint64 max)
{
  gint64 parsed;

  if (!g_ascii_string_to_signed (value, 10, min, max, &parsed, NULL))
    throw std::invalid_argument (
        std::string ("Invalid value of ") + key + ": '" + value + "'");

  return (int32_t) parsed;
}

/**
 * @brief Parse a boolean option value, "true" or "false".
 */
static bool
parse_bool_option (const gchar *key, const gchar *value)
{
  if (g_ascii_strcasecmp (value, "true") == 0)
    return true;
  if (g_ascii_strcasecmp (value, "false") == 0)
    return false;

  throw std::invalid_argument (std::string ("Invalid value of ") + key + ": '"
                               + value + "', true or false is expected");
}

/**
 * @brief Parse the custom property, "key:value" options separated by commas.
 */
void
executorch_llama_subplugin::parseCustomProperties (const char *custom)
{
  g_auto (GStrv) options = NULL;
  guint i;

  if (!custom)
    return;

  options = g_strsplit (custom, ",", -1);
  for (i = 0; options[i]; i++) {
    g_auto (GStrv) option = NULL;
    g_autofree gchar *key = NULL;
    gchar *value;

    if (*g_strstrip (options[i]) == '\0')
      continue;

    option = g_strsplit (options[i], ":", 2);
    if (!option[1])
      throw std::invalid_argument (
          std::string ("Custom option '") + options[i] + "' is not key:value");

    key = g_ascii_strdown (g_strstrip (option[0]), -1);
    value = g_strstrip (option[1]);

    if (g_str_equal (key, "max_new_tokens") || g_str_equal (key, "seq_len")) {
      /* -1 leaves it to the model's context length; 0 generates nothing. */
      int32_t limit = parse_int_option (key, value, -1, G_MAXINT32);
      if (limit == 0)
        throw std::invalid_argument (std::string (key) + " is -1 or positive");
      if (g_str_equal (key, "max_new_tokens"))
        config.max_new_tokens = limit;
      else
        config.seq_len = limit;
    } else if (g_str_equal (key, "temperature")) {
      gchar *end;
      gdouble temperature = g_ascii_strtod (value, &end);

      /* The token generator refuses any other value when it generates. */
      if (end == value || *end != '\0' || !(temperature >= 0.0 && temperature <= 1.0))
        throw std::invalid_argument (
            std::string ("Invalid value of temperature: '") + value + "'");
      config.temperature = (float) temperature;
    } else if (g_str_equal (key, "echo")) {
      config.echo = parse_bool_option (key, value);
    } else if (g_str_equal (key, "ignore_eos")) {
      config.ignore_eos = parse_bool_option (key, value);
    } else if (g_str_equal (key, "keep_eos")) {
      keep_eos = parse_bool_option (key, value);
    } else if (g_str_equal (key, "num_bos")) {
      /* The tokenizer takes the counts as int8_t. */
      config.num_bos = parse_int_option (key, value, 0, G_MAXINT8);
    } else if (g_str_equal (key, "num_eos")) {
      config.num_eos = parse_int_option (key, value, 0, G_MAXINT8);
    } else {
      throw std::invalid_argument (std::string ("Unknown custom option '") + key + "'");
    }
  }
}

/**
 * @brief Method to prepare/configure ExecuTorch instance.
 */
void
executorch_llama_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  try {
    if (prop->num_models != 2)
      throw std::invalid_argument ("Exactly a model (.pte) and a tokenizer are required");

    config.seq_len = 128;
    config.temperature = 0.8f;
    parseCustomProperties (prop->custom_properties);

    /* Load model file (.pte) */
    if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
      const std::string err_msg
          = "Given model " + (std::string) prop->model_files[0] + " is not valid";
      throw std::invalid_argument (err_msg);
    }

    /* Load tokenizer */
    if (!g_file_test (prop->model_files[1], G_FILE_TEST_IS_REGULAR)) {
      const std::string err_msg
          = "Given tokienizer " + (std::string) prop->model_files[1] + " is not valid";
      throw std::invalid_argument (err_msg);
    }

    std::unique_ptr<::tokenizers::Tokenizer> tokenizer
        = llm::load_tokenizer (prop->model_files[1]);
    if (!tokenizer) {
      const std::string err_msg
          = "Failed to load tokenizer " + (std::string) prop->model_files[1];
      throw std::invalid_argument (err_msg);
    }

    /* The runner passes the end-of-sequence token on as text; find that text to leave it out. */
    ::executorch::extension::Module module (
        prop->model_files[0], ::executorch::extension::Module::LoadMode::Mmap);
    for (uint64_t eos : llm::get_eos_ids (tokenizer.get (), &module)) {
      auto text = tokenizer->decode (eos, eos);
      if (text.ok () && !text.get ().empty ())
        eos_texts.insert (text.get ());
    }

    /* set llama runner */
    runner = llm::create_text_llm_runner (prop->model_files[0], std::move (tokenizer));
    if (!runner) {
      const std::string err_msg
          = "Failed to load model " + (std::string) prop->model_files[0];
      throw std::invalid_argument (err_msg);
    }
  } catch (const std::exception &e) {
    /* throw exception upward */
    throw;
  }
}

/**
 * @brief Method to execute the model.
 */
void
executorch_llama_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  UNUSED (input);
  UNUSED (output);

  throw std::runtime_error (
      "executorch-llama only supports invoke-dynamic mode. Set `invoke-dynamic=true`");
}

/**
 * @brief Describe the output as one flexible uint8 tensor of the given size.
 */
static void
set_output_info (GstTensorFilterProperties *prop, guint size)
{
  GstTensorInfo *_info;

  gst_tensors_info_free (&prop->output_meta);

  prop->output_meta.num_tensors = 1;
  prop->output_meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  _info = gst_tensors_info_get_nth_info (&prop->output_meta, 0);
  _info->type = _NNS_UINT8;
  _info->dimension[0] = size;
}

/**
 * @brief Generate text for the prompt. With invoke-async each piece is
 *        dispatched as an output of its own and nothing is returned.
 */
std::string
executorch_llama_subplugin::generate (GstTensorFilterProperties *prop, const std::string &prompt)
{
  std::string result = "";
  bool echoing = config.echo;

  /* Each buffer is a new prompt; do not continue from the previous one. */
  runner->reset ();
  if (runner->generate (prompt, config,
          [this, prop, &result, &echoing] (const std::string &generated) {
            bool is_prompt = echoing;

            /* Before the end-of-sequence check, as a left-out piece is a token too. */
            if (prop->invoke_async && stopping) {
              runner->stop ();
              return;
            }
            /* The echoed prompt comes first and is kept as it is. */
            if (!echoing && !keep_eos && eos_texts.count (generated))
              return;
            echoing = false;
            if (!prop->invoke_async) {
              result.append (generated);
            } else if (is_prompt) {
              /* Held back until a token comes, as the runner may still refuse the prompt. */
              result = generated;
            } else {
              dispatchPiece (prop, result);
              result.clear ();
              dispatchPiece (prop, generated);
            }
          })
      != Error::Ok)
    throw std::runtime_error ("Generating text failed");

  if (prop->invoke_async) {
    if (!stopping)
      dispatchPiece (prop, result);
    result.clear ();
  }

  return result;
}

/**
 * @brief Dispatch a generated piece as an output buffer of its own, unless it is empty.
 */
void
executorch_llama_subplugin::dispatchPiece (
    GstTensorFilterProperties *prop, const std::string &piece)
{
  GstTensorMemory output;

  if (piece.empty ())
    return;

  output.size = piece.size ();
  output.data = g_malloc (output.size);
  /* replace null character with blankspace */
  std::replace_copy (piece.begin (), piece.end (), (char *) output.data, '\0', ' ');

  /* The core takes the size only from here, and its streaming thread reads it unlocked (#5172). */
  gst_tensors_info_get_nth_info (&prop->output_meta, 0)->dimension[0] = output.size;
  nnstreamer_filter_dispatch_output_async (prop, &output);
}

/**
 * @brief Generate the queued prompts in order until the instance is closed.
 */
void
executorch_llama_subplugin::workerLoop ()
{
  std::unique_lock<std::mutex> lock (queue_lock);

  while (true) {
    queue_cond.wait (lock, [this] { return stopping || !prompts.empty (); });
    if (stopping)
      return;

    std::pair<GstTensorFilterProperties *, std::string> next
        = std::move (prompts.front ());
    prompts.pop ();
    lock.unlock ();

    try {
      generate (next.first, next.second);
    } catch (const std::exception &e) {
      ml_loge ("executorch-llama: %s", e.what ());
    } catch (...) {
      ml_loge ("executorch-llama: Generating text failed");
    }

    lock.lock ();
  }
}

/**
 * @brief Method to execute the model.
 */
void
executorch_llama_subplugin::invoke_dynamic (GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");
  if (!input[0].data || input[0].size == 0)
    throw std::invalid_argument ("The prompt is empty.");

  std::string prompt ((char *) input[0].data, input[0].size);

  if (prop->invoke_async) {
    std::lock_guard<std::mutex> lock (queue_lock);

    /* The generated text goes out as dispatched; this invoke itself outputs nothing. */
    if (!worker.joinable ()) {
      set_output_info (prop, 1);
      worker = std::thread (&executorch_llama_subplugin::workerLoop, this);
    }
    prompts.emplace (prop, std::move (prompt));
    queue_cond.notify_one ();
    return;
  }

  std::string result = generate (prop, prompt);
  if (result.empty ())
    throw std::runtime_error ("Generated no text to output");

  /* replace null character with blankspace */
  std::replace (result.begin (), result.end (), '\0', ' ');

  output[0].size = result.size ();
  output[0].data = g_malloc0 (output[0].size);
  memcpy (output[0].data, result.c_str (), result.size ());

  set_output_info (prop, result.size ());
}

/**
 * @brief Method to get the information of ExecuTorch-Llama subplugin.
 */
void
executorch_llama_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = fw_name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 1;
  info.run_without_model = 0;
  info.verify_model_path = 0;
}

/**
 * @brief Method to get the model information.
 */
int
executorch_llama_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    in_info.num_tensors = 1;
    in_info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

    out_info.num_tensors = 1;
    out_info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle events.
 */
int
executorch_llama_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);

  return -ENOENT;
}

executorch_llama_subplugin *executorch_llama_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
executorch_llama_subplugin::init_filter_executorch_llama (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<executorch_llama_subplugin> ();
}

/** @brief Destruct the subplugin */
void
executorch_llama_subplugin::fini_filter_executorch_llama (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief Register the sub-plugin for ExecuTorch-Llama.
 */
void
init_filter_executorch_llama ()
{
  executorch_llama_subplugin::init_filter_executorch_llama ();
}

/**
 * @brief Destruct the sub-plugin for ExecuTorch-Llama.
 */
void
fini_filter_executorch_llama ()
{
  executorch_llama_subplugin::fini_filter_executorch_llama ();
}

} /* namespace tensorfilter_executorch_llama */
} /* namespace nnstreamer */
