/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter, LiteRT Module
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file   tensor_filter_litert.cc
 * @date   20 Aug 2026
 * @brief  LiteRT 2.x (CompiledModel API) module for tensor_filter gstreamer plugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @see    https://github.com/google-ai-edge/LiteRT
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (LiteRT) for tensor_filter.
 * This subplugin targets the LiteRT 2.x "CompiledModel" C API
 * (litert::Environment / Model / CompiledModel / TensorBuffer), which is
 * a different API from the classic TensorFlow-Lite Interpreter API used
 * by the tensorflow2-lite subplugin. Both subplugins may coexist; use
 * framework=litert for this one.
 *
 * The model file consumed is the standard .tflite flatbuffer.
 *
 * Custom properties (tensor_filter custom=...):
 *  - Accelerators:cpu|gpu|npu
 *      Hardware accelerators to enable, combinable with '+'
 *      (e.g., "Accelerators:npu+cpu"). Default: cpu.
 *      The standard "accelerator" property (e.g., accelerator=true:gpu)
 *      is also honored when this custom property is not given.
 *  - Signature:<key>
 *      Select the model signature to run by its key string.
 *      Default: the first signature (index 0).
 *
 * @todo Zero-copy invoke by wrapping GstTensorMemory with
 *       LiteRtCreateTensorBufferFromHostMemory when alignment permits.
 * @todo Support dynamic input dimensions (invoke_dynamic).
 * @todo Expose accelerator-specific opaque options (GPU precision, NPU
 *       compiler plugin paths, etc.).
 */

#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <glib.h>

#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include <litert/c/litert_common.h>
#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_environment.h>
#include <litert/c/litert_layout.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_model_types.h>
#include <litert/c/litert_options.h>
#include <litert/c/litert_tensor_buffer.h>
#include <litert/c/litert_tensor_buffer_requirements.h>
#include <litert/c/litert_tensor_buffer_types.h>

namespace nnstreamer
{
namespace tensorfilter_litert
{
extern "C" {
void _init_filter_litert (void) __attribute__ ((constructor));
void _fini_filter_litert (void) __attribute__ ((destructor));
}

/**
 * @brief Throw std::runtime_error if a LiteRT C API call fails.
 */
#define LITERT_CHECK(expr)                                                         \
  do {                                                                             \
    LiteRtStatus _status = (expr);                                                 \
    if (_status != kLiteRtStatusOk) {                                              \
      ml_loge ("LiteRT error %d at %s", (int) _status, #expr);                     \
      throw std::runtime_error (std::string ("LiteRT call failed (status ")        \
                                + std::to_string ((int) _status) + "): " + #expr); \
    }                                                                              \
  } while (0)

/** @brief litert subplugin class */
class litert_subplugin final : public tensor_filter_subplugin
{
  public:
  static void init_filter_litert ();
  static void fini_filter_litert ();

  litert_subplugin ();
  ~litert_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  private:
  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw = 3;
  static litert_subplugin *registeredRepresentation;

  bool configured{}; /**< Whether this instance has a compiled model. */
  gchar *model_path{}; /**< .tflite model file path */
  LiteRtHwAcceleratorSet accel_set{ kLiteRtHwAcceleratorCpu }; /**< accelerators to request */
  std::string signature_key{}; /**< requested signature key; empty = default */
  LiteRtParamIndex signature_index{}; /**< resolved signature index */

  LiteRtEnvironment env{};
  LiteRtModel model{};
  LiteRtCompiledModel compiled_model{};

  GstTensorsInfo inputTensorMeta;
  GstTensorsInfo outputTensorMeta;

  std::vector<LiteRtTensorBuffer> input_buffers{}; /**< managed, reused per invoke */
  std::vector<LiteRtTensorBuffer> output_buffers{}; /**< managed, reused per invoke */

  void cleanup ();
  void parseCustomProperties (const GstTensorFilterProperties *prop);
  void applyHwList (const GstTensorFilterProperties *prop);
  LiteRtParamIndex resolveSignature () const;
  void setTensorMeta (LiteRtSignature sig, bool is_input, GstTensorsInfo *meta);
  void createTensorBuffers ();

  static tensor_type convertElementType (LiteRtElementType type);
  static void convertLayout (const LiteRtLayout &layout, tensor_dim dim);
  static LiteRtHwAcceleratorSet parseAcceleratorValue (const gchar *value);
};

const char *litert_subplugin::name = "litert";
const accl_hw litert_subplugin::hw_list[] = { ACCL_CPU, ACCL_GPU, ACCL_NPU };
litert_subplugin *litert_subplugin::registeredRepresentation = nullptr;

/**
 * @brief constructor of litert_subplugin
 */
litert_subplugin::litert_subplugin () : tensor_filter_subplugin ()
{
  gst_tensors_info_init (std::addressof (inputTensorMeta));
  gst_tensors_info_init (std::addressof (outputTensorMeta));
}

/**
 * @brief destructor of litert_subplugin
 */
litert_subplugin::~litert_subplugin ()
{
  cleanup ();
}

/**
 * @brief Release all LiteRT objects owned by this instance.
 */
void
litert_subplugin::cleanup ()
{
  for (auto &buf : input_buffers)
    LiteRtDestroyTensorBuffer (buf);
  input_buffers.clear ();
  for (auto &buf : output_buffers)
    LiteRtDestroyTensorBuffer (buf);
  output_buffers.clear ();

  if (compiled_model != nullptr) {
    LiteRtDestroyCompiledModel (compiled_model);
    compiled_model = nullptr;
  }
  if (model != nullptr) {
    LiteRtDestroyModel (model);
    model = nullptr;
  }
  if (env != nullptr) {
    LiteRtDestroyEnvironment (env);
    env = nullptr;
  }

  gst_tensors_info_free (std::addressof (inputTensorMeta));
  gst_tensors_info_free (std::addressof (outputTensorMeta));

  g_free (model_path);
  model_path = nullptr;

  configured = false;
}

/**
 * @brief Method to get an empty object
 */
tensor_filter_subplugin &
litert_subplugin::getEmptyInstance ()
{
  return *(new litert_subplugin ());
}

/**
 * @brief Parse a custom property "Accelerators" value such as "npu+cpu".
 * @return the corresponding LiteRT accelerator bit set
 */
LiteRtHwAcceleratorSet
litert_subplugin::parseAcceleratorValue (const gchar *value)
{
  LiteRtHwAcceleratorSet set = kLiteRtHwAcceleratorNone;
  gchar **entries = g_strsplit_set (value, "+|", -1);

  for (guint i = 0; entries[i] != nullptr; ++i) {
    const gchar *entry = entries[i];

    if (g_ascii_strcasecmp (entry, "cpu") == 0) {
      set |= kLiteRtHwAcceleratorCpu;
    } else if (g_ascii_strcasecmp (entry, "gpu") == 0) {
      set |= kLiteRtHwAcceleratorGpu;
    } else if (g_ascii_strcasecmp (entry, "npu") == 0) {
      set |= kLiteRtHwAcceleratorNpu;
    } else if (g_ascii_strcasecmp (entry, "none") == 0) {
      /* no-op; explicit "let LiteRT decide" */
    } else {
      std::string bad_entry (entry);
      g_strfreev (entries);
      throw std::invalid_argument ("Unknown accelerator \"" + bad_entry
                                   + "\". Supported: cpu, gpu, npu (combinable with '+').");
    }
  }
  g_strfreev (entries);

  return set;
}

/**
 * @brief Map the standard "accelerator" property (prop->hw_list) to LiteRT accelerators.
 */
void
litert_subplugin::applyHwList (const GstTensorFilterProperties *prop)
{
  LiteRtHwAcceleratorSet set = kLiteRtHwAcceleratorNone;

  for (gint i = 0; i < prop->num_hw; ++i) {
    if (prop->hw_list[i] & ACCL_NPU)
      set |= kLiteRtHwAcceleratorNpu;
    else if (prop->hw_list[i] & ACCL_GPU)
      set |= kLiteRtHwAcceleratorGpu;
    else if (prop->hw_list[i] & ACCL_CPU)
      set |= kLiteRtHwAcceleratorCpu;
    else if (prop->hw_list[i] == ACCL_AUTO || prop->hw_list[i] == ACCL_DEFAULT)
      set |= kLiteRtHwAcceleratorCpu;
  }

  if (set != kLiteRtHwAcceleratorNone)
    accel_set = set;
}

/**
 * @brief Parse the custom properties string (e.g., "Accelerators:gpu,Signature:serving_default").
 */
void
litert_subplugin::parseCustomProperties (const GstTensorFilterProperties *prop)
{
  if (!prop->custom_properties)
    return;

  gchar **options = g_strsplit (prop->custom_properties, ",", -1);

  try {
    for (guint i = 0; i < g_strv_length (options); ++i) {
      gchar **option = g_strsplit (options[i], ":", 2);

      if (g_strv_length (option) == 2) {
        g_strstrip (option[0]);
        g_strstrip (option[1]);

        if (g_ascii_strcasecmp (option[0], "Accelerators") == 0) {
          accel_set = parseAcceleratorValue (option[1]);
        } else if (g_ascii_strcasecmp (option[0], "Signature") == 0) {
          signature_key = option[1];
        } else {
          ml_logw ("Unknown custom property [%s]. This is ignored.", option[0]);
        }
      } else if (option[0] != nullptr && option[0][0] != '\0') {
        g_strfreev (option);
        throw std::invalid_argument (
            std::string ("Malformed custom property \"") + options[i]
            + "\". Expected Key:Value pairs separated by ','.");
      }
      g_strfreev (option);
    }
  } catch (...) {
    g_strfreev (options);
    throw;
  }
  g_strfreev (options);
}

/**
 * @brief Convert a LiteRT element type to nnstreamer tensor_type.
 */
tensor_type
litert_subplugin::convertElementType (LiteRtElementType type)
{
  switch (type) {
    case kLiteRtElementTypeFloat32:
      return _NNS_FLOAT32;
    case kLiteRtElementTypeFloat64:
      return _NNS_FLOAT64;
    case kLiteRtElementTypeFloat16:
      return _NNS_FLOAT16;
    case kLiteRtElementTypeInt8:
      return _NNS_INT8;
    case kLiteRtElementTypeInt16:
      return _NNS_INT16;
    case kLiteRtElementTypeInt32:
      return _NNS_INT32;
    case kLiteRtElementTypeInt64:
      return _NNS_INT64;
    case kLiteRtElementTypeUInt8:
    case kLiteRtElementTypeBool:
      return _NNS_UINT8;
    case kLiteRtElementTypeUInt16:
      return _NNS_UINT16;
    case kLiteRtElementTypeUInt32:
      return _NNS_UINT32;
    case kLiteRtElementTypeUInt64:
      return _NNS_UINT64;
    default:
      throw std::invalid_argument (std::string ("Unsupported LiteRT element type: ")
                                   + std::to_string ((int) type));
  }
}

/**
 * @brief Convert a LiteRT layout to nnstreamer tensor_dim (reversed order).
 */
void
litert_subplugin::convertLayout (const LiteRtLayout &layout, tensor_dim dim)
{
  guint rank = layout.rank;

  for (guint i = 0; i < NNS_TENSOR_RANK_LIMIT; ++i)
    dim[i] = 0;

  if (rank > NNS_TENSOR_RANK_LIMIT)
    throw std::invalid_argument (std::string ("Tensor rank ") + std::to_string (rank) + " exceeds the limit ("
                                 + std::to_string (NNS_TENSOR_RANK_LIMIT) + ").");

  if (rank == 0) {
    /* scalar; represent as a single-element tensor */
    dim[0] = 1;
    return;
  }

  /* the order of dimension is reversed at CAPS negotiation */
  for (guint i = 0; i < rank; ++i) {
    int32_t d = layout.dimensions[rank - 1 - i];

    if (d < 0)
      throw std::invalid_argument (
          "Dynamic dimensions are not supported yet by the litert subplugin.");
    dim[i] = (uint32_t) d;
  }
}

/**
 * @brief Resolve the signature to run; by key if given, otherwise index 0.
 */
LiteRtParamIndex
litert_subplugin::resolveSignature () const
{
  LiteRtParamIndex num_signatures = 0;

  LITERT_CHECK (LiteRtGetNumModelSignatures (model, &num_signatures));
  if (num_signatures == 0)
    throw std::runtime_error ("The model does not have any signature to run.");

  if (signature_key.empty ())
    return 0;

  for (LiteRtParamIndex i = 0; i < num_signatures; ++i) {
    LiteRtSignature sig = nullptr;
    const char *key = nullptr;

    LITERT_CHECK (LiteRtGetModelSignature (model, i, &sig));
    LITERT_CHECK (LiteRtGetSignatureKey (sig, &key));
    if (key != nullptr && signature_key == key)
      return i;
  }

  throw std::invalid_argument (std::string ("Signature \"") + signature_key
                               + "\" is not found in the model.");
}

/**
 * @brief Fill a GstTensorsInfo from the signature's input or output tensors.
 *
 * The element type and name come from the model signature; the dimensions
 * come from the compiled model's resolved layouts so that dynamic dimensions
 * declared in the model (e.g., batch = -1) are resolved to the concrete
 * shapes allocated by the runtime, matching the classic Interpreter API
 * behavior after AllocateTensors().
 */
void
litert_subplugin::setTensorMeta (LiteRtSignature sig, bool is_input, GstTensorsInfo *meta)
{
  size_t num_tensors = 0;
  std::vector<LiteRtLayout> out_layouts;

  if (is_input) {
    LITERT_CHECK (LiteRtGetNumSignatureInputs (sig, &num_tensors));
  } else {
    LITERT_CHECK (LiteRtGetNumSignatureOutputs (sig, &num_tensors));
    out_layouts.resize (num_tensors);
    LITERT_CHECK (LiteRtGetCompiledModelOutputTensorLayouts (compiled_model,
        signature_index, num_tensors, out_layouts.data (), false));
  }

  if (num_tensors > NNS_TENSOR_SIZE_LIMIT)
    throw std::invalid_argument (
        std::string ("The number of ") + (is_input ? "input" : "output")
        + " tensors (" + std::to_string (num_tensors) + ") exceeds the limit ("
        + std::to_string (NNS_TENSOR_SIZE_LIMIT) + ").");

  meta->num_tensors = (unsigned int) num_tensors;

  for (size_t i = 0; i < num_tensors; ++i) {
    LiteRtTensor tensor = nullptr;
    const char *tensor_name = nullptr;
    LiteRtTensorTypeId type_id;
    LiteRtRankedTensorType ranked_type;
    LiteRtLayout layout;
    GstTensorInfo *info = gst_tensors_info_get_nth_info (meta, (guint) i);

    if (is_input) {
      LITERT_CHECK (LiteRtGetSignatureInputName (sig, i, &tensor_name));
      LITERT_CHECK (LiteRtGetSignatureInputTensorByIndex (sig, i, &tensor));
      LITERT_CHECK (LiteRtGetCompiledModelInputTensorLayout (
          compiled_model, signature_index, i, &layout));
    } else {
      LITERT_CHECK (LiteRtGetSignatureOutputName (sig, i, &tensor_name));
      LITERT_CHECK (LiteRtGetSignatureOutputTensorByIndex (sig, i, &tensor));
      layout = out_layouts[i];
    }

    LITERT_CHECK (LiteRtGetTensorTypeId (tensor, &type_id));
    if (type_id != kLiteRtRankedTensorType)
      throw std::invalid_argument ("Tensors with dynamic (unranked) types are not supported.");

    LITERT_CHECK (LiteRtGetRankedTensorType (tensor, &ranked_type));

    info->type = convertElementType (ranked_type.element_type);
    convertLayout (layout, info->dimension);
    info->name = g_strdup (tensor_name);

    ml_logd ("litert %s tensorMeta[%zu] >> name[%s], type[%d]",
        is_input ? "input" : "output", i, info->name ? info->name : "(null)", info->type);
  }
}

/**
 * @brief Create managed tensor buffers for all inputs/outputs, reused across invokes.
 */
void
litert_subplugin::createTensorBuffers ()
{
  LiteRtSignature sig = nullptr;
  std::vector<LiteRtLayout> out_layouts (outputTensorMeta.num_tensors);
  guint i;

  LITERT_CHECK (LiteRtGetModelSignature (model, signature_index, &sig));
  LITERT_CHECK (LiteRtGetCompiledModelOutputTensorLayouts (compiled_model,
      signature_index, out_layouts.size (), out_layouts.data (), false));

  for (i = 0; i < inputTensorMeta.num_tensors; ++i) {
    LiteRtTensorBufferRequirements reqs = nullptr;
    LiteRtRankedTensorType ranked_type;
    LiteRtTensor tensor = nullptr;
    LiteRtTensorBuffer buf = nullptr;

    LITERT_CHECK (LiteRtGetSignatureInputTensorByIndex (sig, i, &tensor));
    LITERT_CHECK (LiteRtGetRankedTensorType (tensor, &ranked_type));
    /* use the runtime-resolved layout; the model may declare dynamic dims */
    LITERT_CHECK (LiteRtGetCompiledModelInputTensorLayout (
        compiled_model, signature_index, i, &ranked_type.layout));
    LITERT_CHECK (LiteRtGetCompiledModelInputBufferRequirements (
        compiled_model, signature_index, i, &reqs));
    LITERT_CHECK (LiteRtCreateManagedTensorBufferFromRequirements (
        env, &ranked_type, reqs, &buf));
    input_buffers.push_back (buf);
  }

  for (i = 0; i < outputTensorMeta.num_tensors; ++i) {
    LiteRtTensorBufferRequirements reqs = nullptr;
    LiteRtRankedTensorType ranked_type;
    LiteRtTensor tensor = nullptr;
    LiteRtTensorBuffer buf = nullptr;

    LITERT_CHECK (LiteRtGetSignatureOutputTensorByIndex (sig, i, &tensor));
    LITERT_CHECK (LiteRtGetRankedTensorType (tensor, &ranked_type));
    ranked_type.layout = out_layouts[i];
    LITERT_CHECK (LiteRtGetCompiledModelOutputBufferRequirements (
        compiled_model, signature_index, i, &reqs));
    LITERT_CHECK (LiteRtCreateManagedTensorBufferFromRequirements (
        env, &ranked_type, reqs, &buf));
    output_buffers.push_back (buf);
  }
}

/**
 * @brief Configure the instance: load and compile the model, gather tensor info.
 */
void
litert_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  LiteRtOptions options = nullptr;
  LiteRtSignature sig = nullptr;

  if (prop->num_models != 1 || !prop->model_files[0] || prop->model_files[0][0] == '\0') {
    ml_loge ("LiteRT filter requires one .tflite model file.");
    throw std::invalid_argument ("The .tflite model file is not given.");
  }

  if (configured)
    cleanup ();

  gst_tensors_info_init (std::addressof (inputTensorMeta));
  gst_tensors_info_init (std::addressof (outputTensorMeta));

  model_path = g_strdup (prop->model_files[0]);

  applyHwList (prop);
  try {
    parseCustomProperties (prop);
  } catch (const std::invalid_argument &e) {
    cleanup ();
    ml_loge ("Failed to parse custom property: %s", e.what ());
    throw std::invalid_argument (
        "Failed to parse custom property: " + std::string (e.what ()));
  }

  try {
    LITERT_CHECK (LiteRtCreateEnvironment (0, nullptr, &env));
    LITERT_CHECK (LiteRtCreateModelFromFile (env, model_path, &model));

    signature_index = resolveSignature ();
    LITERT_CHECK (LiteRtGetModelSignature (model, signature_index, &sig));

    LITERT_CHECK (LiteRtCreateOptions (&options));
    LITERT_CHECK (LiteRtSetOptionsHardwareAccelerators (options, accel_set));
    {
      LiteRtStatus status = LiteRtCreateCompiledModel (env, model, options, &compiled_model);
      LiteRtDestroyOptions (options);
      options = nullptr;
      if (status != kLiteRtStatusOk) {
        ml_loge ("Failed to compile the LiteRT model %s (status %d). "
                 "Check whether the requested accelerator is available.",
            model_path, (int) status);
        throw std::runtime_error ("LiteRtCreateCompiledModel failed.");
      }
    }

    /* tensor meta needs the compiled model for runtime-resolved layouts */
    setTensorMeta (sig, true, std::addressof (inputTensorMeta));
    setTensorMeta (sig, false, std::addressof (outputTensorMeta));

    createTensorBuffers ();
  } catch (...) {
    if (options != nullptr)
      LiteRtDestroyOptions (options);
    cleanup ();
    throw;
  }

  configured = true;
}

/**
 * @brief Invoke the model with the given input and output tensors.
 */
void
litert_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  guint i;

  if (!configured)
    throw std::runtime_error ("Invoke called before a model is configured.");

  if (input == nullptr || output == nullptr)
    throw std::invalid_argument ("Invoke called with a null tensor memory.");

  /* Fill input buffers */
  for (i = 0; i < inputTensorMeta.num_tensors; ++i) {
    void *host_mem = nullptr;

    if (input[i].data == nullptr)
      throw std::invalid_argument ("Input tensor memory is null.");

    LITERT_CHECK (LiteRtLockTensorBuffer (
        input_buffers[i], &host_mem, kLiteRtTensorBufferLockModeWrite));
    std::memcpy (host_mem, input[i].data, input[i].size);
    LITERT_CHECK (LiteRtUnlockTensorBuffer (input_buffers[i]));
  }

  LITERT_CHECK (LiteRtRunCompiledModel (compiled_model, signature_index,
      input_buffers.size (), input_buffers.data (), output_buffers.size (),
      output_buffers.data ()));

  /* Read back output buffers */
  for (i = 0; i < outputTensorMeta.num_tensors; ++i) {
    void *host_mem = nullptr;

    if (output[i].data == nullptr)
      throw std::invalid_argument ("Output tensor memory is null.");

    LITERT_CHECK (LiteRtLockTensorBuffer (
        output_buffers[i], &host_mem, kLiteRtTensorBufferLockModeRead));
    std::memcpy (output[i].data, host_mem, output[i].size);
    LITERT_CHECK (LiteRtUnlockTensorBuffer (output_buffers[i]));
  }
}

/**
 * @brief Describe the framework to the tensor_filter infrastructure.
 */
void
litert_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = FALSE;
  info.allocate_in_invoke = FALSE;
  info.run_without_model = FALSE;
  info.verify_model_path = TRUE;
  info.hw_list = hw_list;
  info.num_hw = num_hw;
}

/**
 * @brief Get the in/out tensors info of the configured model.
 */
int
litert_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops != GET_IN_OUT_INFO)
    return -ENOENT;

  if (!configured)
    return -EINVAL;

  gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputTensorMeta));
  gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputTensorMeta));

  return 0;
}

/**
 * @brief Handle tensor_filter framework events.
 */
int
litert_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

/**
 * @brief Register the litert_subplugin object.
 */
void
litert_subplugin::init_filter_litert (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<litert_subplugin> ();
  nnstreamer_filter_set_custom_property_desc (name, "Accelerators",
      "Hardware accelerators to enable: cpu, gpu, npu; combinable with '+' "
      "(e.g., \"npu+cpu\"). Default: cpu",
      "Signature",
      "Key of the model signature to run (default: the first signature)", NULL);
}

/**
 * @brief Unregister the litert_subplugin object.
 */
void
litert_subplugin::fini_filter_litert (void)
{
  g_assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
_init_filter_litert (void)
{
  litert_subplugin::init_filter_litert ();
}

/** @brief Destruct the subplugin */
void
_fini_filter_litert (void)
{
  litert_subplugin::fini_filter_litert ();
}

} /* namespace tensorfilter_litert */
} /* namespace nnstreamer */
