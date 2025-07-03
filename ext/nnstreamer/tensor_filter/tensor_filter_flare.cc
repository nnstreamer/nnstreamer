/**
 * GStreamer Tensor_Filter, flare Module
 * Copyright (C) 2025 Hyunil Park <hyunil46.park@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file    tensor_filter_flare.cc
 * @date    2 July 2025
 * @brief   flare module for tensor_filter gstreamer plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Hyunil Park <hyunil46.park@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <SFlareApi.h>
#include <assert.h>
#include <atomic>
#include <condition_variable>
#include <errno.h>
#include <functional>
#include <iostream>
#include <memory>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_filter_flare (void) __attribute__ ((constructor));
void fini_filter_flare (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

namespace nnstreamer
{
namespace tensorfilter_flare
{
/**
 * @brief Class for TensorFilterFlare subplugin
 */
class TensorFilterFlare : public tensor_filter_subplugin
{
  public:
  TensorFilterFlare ();
  ~TensorFilterFlare ();

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
  static void init_filter_flare ();
  static void fini_filter_flare ();

  private:
  static TensorFilterFlare *registered;
  static const char *name;

  SFlareApi::SFlareContext *context;
  SFlareApi::SFlareConfig config;
  SFlareApi::BackendType backend;
  std::string tokenizer_path;
  bool enable_fsu;
  size_t output_size;
  bool is_configured;

  void parseCustomProperties (const GstTensorFilterProperties *prop);
  void initializeFlareContext ();
};

TensorFilterFlare *TensorFilterFlare::registered = nullptr;
const char *TensorFilterFlare::name = "flare";

/**
 * @brief Construct a new flare subplugin instance
 */
TensorFilterFlare::TensorFilterFlare ()
    : context (nullptr), backend (SFlareApi::BackendType::CPU),
      enable_fsu (false), output_size (1024), is_configured (false), config{}
{
}

/**
 * @brief Destructor of TensorFilterFlare
 */
TensorFilterFlare::~TensorFilterFlare ()
{
  if (context) {
    SFlareApi::DestroySFlareContext (context);
    context = nullptr;
  }
}

/**
 * @brief Method to get an empty object
 */
tensor_filter_subplugin &
TensorFilterFlare::getEmptyInstance ()
{
  return *(new TensorFilterFlare ());
}

/**
 * @brief Parse custom prop and set instance options accordingly.
 */
void
TensorFilterFlare::parseCustomProperties (const GstTensorFilterProperties *prop)
{
  using uniq_g_strv = std::unique_ptr<gchar *, std::function<void (gchar **)>>;
  guint len, i;
  std::string model_type;
  std::string data_type;
  std::string backend_type;

  if (!prop->custom_properties) {
    throw std::invalid_argument (
        "Custom properties are required for FLARE filter. Please provide tokenizer_path, backend, model_type, and data_type.");
    return;
  }

  uniq_g_strv options (g_strsplit (prop->custom_properties, ",", -1), g_strfreev);
  len = g_strv_length (options.get ());

  for (i = 0; i < len; ++i) {
    uniq_g_strv option (g_strsplit (options.get ()[i], ":", -1), g_strfreev);

    if (g_strv_length (option.get ()) > 1) {
      g_strstrip (option.get ()[0]);
      g_strstrip (option.get ()[1]);

      if (g_ascii_strcasecmp (option.get ()[0], "tokenizer_path") == 0) {
        tokenizer_path = option.get ()[1];
        config.tokenizer_path = tokenizer_path.c_str ();
      } else if (g_ascii_strcasecmp (option.get ()[0], "backend") == 0) {
        if (g_ascii_strcasecmp (option.get ()[1], "gpu") == 0) {
          backend = SFlareApi::BackendType::GPU;
          backend_type = "GPU";
        } else if (g_ascii_strcasecmp (option.get ()[1], "cpu") == 0) {
          backend = SFlareApi::BackendType::CPU;
          backend_type = "CPU";
        } else {
          throw std::invalid_argument (
              "Invalid backend value: " + std::string (option.get ()[1]));
        }
      } else if (g_ascii_strcasecmp (option.get ()[0], "output_size") == 0) {
        output_size
            = static_cast<size_t> (g_ascii_strtoull (option.get ()[1], NULL, 10));
      } else if (g_ascii_strcasecmp (option.get ()[0], "model_type") == 0) {
        if (g_ascii_strcasecmp (option.get ()[1], "1B") == 0) {
          config.llm_model = SFlareApi::GaussFModelType::GAUSS1B;
          model_type = "GAUSS1B";
        } else if (g_ascii_strcasecmp (option.get ()[1], "3B") == 0) {
          config.llm_model = SFlareApi::GaussFModelType::GAUSS3B;
          model_type = "GAUSS3B";
        } else {
          throw std::invalid_argument (
              "Invalid model_type value: " + std::string (option.get ()[1]));
        }
      } else if (g_ascii_strcasecmp (option.get ()[0], "data_type") == 0) {
        if (g_ascii_strcasecmp (option.get ()[1], "W4A32") == 0) {
          config.data_type = SFlareApi::ModelDataType::DTYPE_W4A32;
          data_type = "DTYPE_W4A32";
        } else if (g_ascii_strcasecmp (option.get ()[1], "W8A32") == 0) {
          config.data_type = SFlareApi::ModelDataType::DTYPE_W8A32;
          data_type = "DTYPE_W8A32";
        } else if (g_ascii_strcasecmp (option.get ()[1], "W16A32") == 0) {
          config.data_type = SFlareApi::ModelDataType::DTYPE_W16A32;
          data_type = "DTYPE_W16A32";
        } else if (g_ascii_strcasecmp (option.get ()[1], "W32A32") == 0) {
          config.data_type = SFlareApi::ModelDataType::DTYPE_W32A32;
          data_type = "DTYPE_W32A32";
        } else {
          throw std::invalid_argument (
              "Invalid data_type value: " + std::string (option.get ()[1]));
        }
      } else if (g_ascii_strcasecmp (option.get ()[0], "enable_fsu") == 0) {
        if (g_ascii_strcasecmp (option.get ()[1], "true") == 0) {
          enable_fsu = true;
        } else if (g_ascii_strcasecmp (option.get ()[1], "false") == 0) {
          enable_fsu = false;
        } else {
          throw std::invalid_argument (
              "Invalid enable_fsu value: " + std::string (option.get ()[1]));
        }
      } else {
        throw std::invalid_argument (
            "Unknown custom property " + std::string (option.get ()[0]));
      }
    }
  }

  ml_logi ("FLARE Config - tokenizer_path: %s, backend: %s, output_size: %zu, model_type: %s, data_type: %s",
      config.tokenizer_path, backend_type.c_str (), output_size,
      model_type.c_str (), data_type.c_str ());

  /* Validate required configuration */
  if (tokenizer_path.empty ()) {
    throw std::invalid_argument ("tokenizer_path is required");
  }

  is_configured = true;
  return;
}

/**
 * @brief Initialize flare context with given configuration.
 */
void
TensorFilterFlare::initializeFlareContext ()
{
  if (!is_configured) {
    throw std::runtime_error (
        "FLARE context not properly configured. Call parseCustomProperties first.");
  }

  context = SFlareApi::initSFlare ();
  if (!context) {
    throw std::runtime_error ("Failed to initialize flare context");
  }

  SFlareApi::ErrorCode result;
  result = context->setSFlareOptions (config);
  if (result != SFlareApi::ErrorCode::SFLARE_SUCCESS) {
    SFlareApi::DestroySFlareContext (context);
    context = nullptr;
    throw std::runtime_error ("Failed to set flare options");
  }

  result = context->loadSFlareLLMModel (backend, enable_fsu);
  if (result != SFlareApi::ErrorCode::SFLARE_SUCCESS) {
    SFlareApi::DestroySFlareContext (context);
    context = nullptr;
    throw std::runtime_error ("Failed to load flare model");
  }
}

/**
 * @brief Configure flare instance
 */
void
TensorFilterFlare::configure_instance (const GstTensorFilterProperties *prop)
{
  if (!prop->invoke_dynamic) {
    throw std::invalid_argument (
        "Flare only supports invoke-dynamic mode. Set `invoke-dynamic=true`");
  }

  try {
    parseCustomProperties (prop);
  } catch (const std::invalid_argument &e) {
    throw std::invalid_argument ("Failed to parse \"custom\" prop:"
                                 + std::string (e.what ()) + "\n\tReference: " + __FILE__);
  }
  try {
    initializeFlareContext ();
  } catch (const std::exception &e) {
    throw std::runtime_error (
        "Failed to initialize flare context: " + std::string (e.what ()));
  }

  ml_logd ("model path(%s) is not used. The model path is hardcoded within the API.",
      prop->model_files[0] ? prop->model_files[0] : "<NULL>");
}

/**
 * @brief Invoke flare using input tensors
 */
void
TensorFilterFlare::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  UNUSED (input);
  UNUSED (output);
  throw std::runtime_error ("Flare only supports invoke-dynamic mode. Set `invoke-dynamic=true`");
}

/**
 * @brief Invoke flare using input tensors
 */
void
TensorFilterFlare::invoke_dynamic (GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  std::string prompt;
  std::string output_text;
  SFlareApi::ErrorCode result;

  if (!context) {
    throw std::runtime_error ("FLARE context not initialized. Call configure_instance first.");
  }

  if (!input || !input[0].data || input[0].size == 0) {
    throw std::invalid_argument ("Invalid input tensor data");
  }

  /* Ensure input is null-terminated for string processing */
  std::string input_str;
  try {
    input_str.assign ((char *) input[0].data, input[0].size);
  } catch (const std::exception &e) {
    throw std::runtime_error (
        "Failed to process input tensor data: " + std::string (e.what ()));
  }

  result = context->executeSFlareLLM (input_str, output_text, output_size);
  if (result != SFlareApi::ErrorCode::SFLARE_SUCCESS) {
    throw std::runtime_error ("Failed to execute sflare LLM");
  }

  /* Stores string length, excluding NULL appended to the end of the string */
  output[0].size = output_text.length ();
  output[0].data = strndup (output_text.c_str (), output_text.length ());
  if (!output[0].data) {
    throw std::runtime_error ("Failed to allocate memory for output tensor");
  }
  g_print ("%s", (char *) output[0].data);

  gst_tensors_info_free (&prop->output_meta);
  prop->output_meta.num_tensors = 1;
  prop->output_meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  prop->output_meta.info[0].type = _NNS_UINT8;
  prop->output_meta.info[0].dimension[0] = output[0].size;
}

/**
 * @brief Get framework info.
 */
void
TensorFilterFlare::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 1;
  info.run_without_model = 0;
  info.verify_model_path = 1;
}

/**
 * @brief Get model info.
 */
int
TensorFilterFlare::getModelInfo (
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
TensorFilterFlare::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

/**
 * @brief Initialize this object for tensor_filter subplugin runtime register
 */
void
TensorFilterFlare::init_filter_flare ()
{
  registered = tensor_filter_subplugin::register_subplugin<TensorFilterFlare> ();
  nnstreamer_filter_set_custom_property_desc (name, "tokenizer_path",
      "Path to the tokenizer vocabulary file", "backend", "LLM backend type: 'cpu' or 'gpu'",
      "output_size", "Number of characters to generate as output", "model_type",
      "Model type: 'GAUSS31', 'GAUSS3B'", "data_type",
      "Data type: 'W4A32', 'W8A32', 'W16A32', 'W32A32'", "enable_fsu",
      "run FSU (Flash Storage Utilization)", NULL);
}

/** @brief Destruct the subplugin */
void
TensorFilterFlare::fini_filter_flare ()
{
  /* internal logic error */
  assert (registered != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registered);
}

} /* namespace tensorfilter_flare */
} /* namespace nnstreamer */

/**
 * @brief Subplugin initializer
 */
void
init_filter_flare ()
{
  nnstreamer::tensorfilter_flare::TensorFilterFlare::init_filter_flare ();
}

/**
 * @brief Subplugin finalizer
 */
void
fini_filter_flare ()
{
  nnstreamer::tensorfilter_flare::TensorFilterFlare::fini_filter_flare ();
}
