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
 * @todo Zero-copy the input side too. Outputs already skip the copy when
 *       LiteRT accepts the caller's memory, but inputs cannot: tensor_filter
 *       maps them GST_MAP_READ and the memory may be shared with another
 *       branch, so it would take a mapping change in the element plus a
 *       statement from upstream that a run never writes to an input.
 * @todo Support dynamic input dimensions (invoke_dynamic).
 * @todo Expose accelerator-specific opaque options (GPU precision, NPU
 *       compiler plugin paths, etc.).
 */

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
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

namespace
{
/**
 * @brief The process-wide LiteRtEnvironment shared by every filter instance.
 *
 * LiteRT binds accelerator device contexts (the GPU context, the dispatch and
 * compiler-plugin libraries) to the environment, and upstream recommends
 * sharing one environment when an application holds several compiled models.
 * Nothing in it is per-instance here: accelerator selection is applied through
 * LiteRtCreateOptions at compile time, not through the environment.
 *
 * It is reference counted rather than leaked as a create-once singleton
 * because the environment must outlive every model and compiled model created
 * from it, while instances open and close independently at pipeline state
 * changes.
 *
 * The lock is shared/exclusive because the two access patterns differ.
 * Building and tearing down an instance's LiteRT object graph takes it
 * exclusively: before the environment was shared each instance built its own,
 * so those paths could not interact, and LiteRT documents no contract that
 * lets them interact now. invoke() takes it in shared mode instead, so
 * inference is not serialized across instances.
 *
 * Note what upstream does and does not say. litert/cc/litert_environment.h
 * ("In a case of having multiple CompiledModels, it is recommended to share
 * the same Environment") sanctions the structure, not concurrent calls into
 * it, and litert/cc/litert_compiled_model.h requires only that the
 * environment outlive the compiled model and any execution running on it.
 * Neither states a thread-safety contract. Concurrent invoke is therefore a
 * reasoned choice, not a documented guarantee: serializing every inference in
 * the process would be a far worse regression, and libLiteRt.so is a prebuilt
 * binary, so no sanitizer can settle the question either. If environment-level
 * state ever does prove unsafe under concurrent use, it will show up on
 * GPU/NPU rather than on the CPU path CI exercises, and this is the comment to
 * come back to.
 *
 * The costs: instances are no longer fully independent, since configuring or
 * tearing one down blocks invokes on all the others and an invoke in flight
 * delays another instance's setup - both bounded by a single model compile on
 * a path that only runs at pipeline state changes. Every instance in the
 * process also shares this lock's cache line, so with enough of them even
 * readers serialize on that atomic.
 */
std::shared_mutex litert_env_lock;
LiteRtEnvironment litert_env = nullptr;
unsigned int litert_env_refs = 0;

/**
 * @brief Output size from which wrapping the caller's memory beats copying it.
 *
 * Wrapping is not free: creating and destroying a tensor buffer around caller
 * memory measured 1.3-2.4 us on x86_64, against 0.06-0.28 us to lock, copy and
 * unlock a 4 kB output. On a classifier it therefore costs an order of
 * magnitude more than the copy it removes, and only starts paying once the
 * copy grows. A segmentation model's 5 MB output, by contrast, copies for
 * hundreds of microseconds and wraps for the same 2 us.
 *
 * Working the break-even back from the large case, where the copy is memory
 * bound rather than cache resident, puts it near 26-49 kB. 256 kB therefore
 * leaves five to ten times the margin, so a tensor taking the direct path
 * wins clearly rather than marginally. The figures are indicative of one
 * machine; the threshold is deliberately conservative so that being wrong
 * about them costs a copy that was already cheap.
 */
constexpr size_t litert_wrap_min_bytes = 256 * 1024;

/**
 * @brief Whether a pointer meets LiteRT's host memory buffer alignment.
 */
inline bool
isHostMemoryAligned (const void *data)
{
  return (reinterpret_cast<uintptr_t> (data) % LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) == 0;
}

/**
 * @brief Destroys the tensor buffers that wrap caller memory for one invoke.
 *
 * The wrappers are created with a null deallocator, so this releases only the
 * LiteRT objects and never the caller's memory. Holding them in a scope guard
 * keeps a throw between creation and the run from leaking them.
 */
class wrapped_buffers final
{
  public:
  wrapped_buffers () = default;
  ~wrapped_buffers ()
  {
    for (auto &buf : held)
      LiteRtDestroyTensorBuffer (buf);
  }

  wrapped_buffers (const wrapped_buffers &) = delete;
  wrapped_buffers &operator= (const wrapped_buffers &) = delete;

  /** @brief Take ownership of one wrapper. */
  void hold (LiteRtTensorBuffer buf)
  {
    held.push_back (buf);
  }

  private:
  std::vector<LiteRtTensorBuffer> held{};
};

/**
 * @brief Take a reference to the shared environment, creating it if needed.
 * @return the shared environment; the caller must not destroy it
 * @throw std::runtime_error if the environment cannot be created
 */
LiteRtEnvironment
litert_env_ref ()
{
  std::lock_guard<std::shared_mutex> guard (litert_env_lock);

  if (litert_env_refs == 0) {
    LiteRtEnvironment created = nullptr;
    LITERT_CHECK (LiteRtCreateEnvironment (0, nullptr, &created));
    litert_env = created;
  }

  ++litert_env_refs;
  return litert_env;
}

/**
 * @brief Drop a reference, destroying the environment along with the last one.
 */
void
litert_env_unref ()
{
  std::lock_guard<std::shared_mutex> guard (litert_env_lock);

  if (litert_env_refs == 0) {
    /** Defensive, but never legitimate: it means some path released a
     *  reference it did not hold, so say so instead of absorbing it. */
    ml_loge ("Unbalanced LiteRT environment release; this is a bug.");
    return;
  }

  if (--litert_env_refs > 0)
    return;

  LiteRtDestroyEnvironment (litert_env);
  litert_env = nullptr;
}
} /* namespace */

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

  LiteRtEnvironment env{}; /**< borrowed from the shared environment; never destroyed here */
  LiteRtModel model{};
  LiteRtCompiledModel compiled_model{};

  GstTensorsInfo inputTensorMeta;
  GstTensorsInfo outputTensorMeta;

  std::vector<LiteRtTensorBuffer> input_buffers{}; /**< managed, reused per invoke */
  std::vector<LiteRtTensorBuffer> output_buffers{}; /**< managed, reused per invoke */
  std::vector<size_t> input_tensor_sizes{}; /**< nnstreamer tensor sizes; each is <= its LiteRT buffer */
  std::vector<size_t> output_tensor_sizes{}; /**< nnstreamer tensor sizes; each is <= its LiteRT buffer */

  std::vector<LiteRtRankedTensorType> output_types{}; /**< to wrap caller memory per invoke */
  std::vector<char> output_wrappable{}; /**< requirements allow host memory and the sizes match exactly */
  std::vector<LiteRtTensorBuffer> invoke_outputs{}; /**< per-invoke argument array, storage reused */

  void cleanup ();
  void parseCustomProperties (const GstTensorFilterProperties *prop);
  void applyHwList (const GstTensorFilterProperties *prop);
  LiteRtParamIndex resolveSignature () const;
  void setTensorMeta (LiteRtSignature sig, bool is_input, GstTensorsInfo *meta);
  void createTensorBuffers ();
  static bool requirementsAllowHostMemory (LiteRtTensorBufferRequirements reqs);

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
  {
    std::lock_guard<std::shared_mutex> guard (litert_env_lock);

    for (auto &buf : input_buffers)
      LiteRtDestroyTensorBuffer (buf);
    input_buffers.clear ();
    input_tensor_sizes.clear ();
    for (auto &buf : output_buffers)
      LiteRtDestroyTensorBuffer (buf);
    output_buffers.clear ();
    output_tensor_sizes.clear ();
    output_types.clear ();
    output_wrappable.clear ();
    invoke_outputs.clear ();

    if (compiled_model != nullptr) {
      LiteRtDestroyCompiledModel (compiled_model);
      compiled_model = nullptr;
    }
    if (model != nullptr) {
      LiteRtDestroyModel (model);
      model = nullptr;
    }
  }

  if (env != nullptr) {
    litert_env_unref ();
    env = nullptr;
  }

  gst_tensors_info_free (std::addressof (inputTensorMeta));
  gst_tensors_info_free (std::addressof (outputTensorMeta));

  g_free (model_path);
  model_path = nullptr;

  /** restore the property defaults so a re-configured instance does not
   *  inherit the previous model's accelerator or signature selection */
  accel_set = kLiteRtHwAcceleratorCpu;
  signature_key.clear ();
  signature_index = 0;

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

    if (entry[0] == '\0') {
      /* tolerate empty entries from consecutive/trailing delimiters ("cpu++gpu") */
      continue;
    }

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
    case kLiteRtElementTypeBool: /* nnstreamer has no bool; both are 1-byte */
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

  if (layout.has_strides)
    throw std::invalid_argument (
        "Strided (non-contiguous) tensor layouts are not supported yet by the litert subplugin.");

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
  }

  if (num_tensors == 0)
    throw std::invalid_argument (std::string ("The model signature has no ")
                                 + (is_input ? "input" : "output") + " tensor.");

  if (num_tensors > NNS_TENSOR_SIZE_LIMIT)
    throw std::invalid_argument (
        std::string ("The number of ") + (is_input ? "input" : "output")
        + " tensors (" + std::to_string (num_tensors) + ") exceeds the limit ("
        + std::to_string (NNS_TENSOR_SIZE_LIMIT) + ").");

  if (!is_input) {
    out_layouts.resize (num_tensors);
    LITERT_CHECK (LiteRtGetCompiledModelOutputTensorLayouts (compiled_model,
        signature_index, num_tensors, out_layouts.data (), false));
  }

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

    /* the buffer LiteRT allocated must be able to hold the whole tensor */
    size_t buf_size = 0;
    LITERT_CHECK (LiteRtGetTensorBufferSize (buf, &buf_size));
    gsize nns_size
        = gst_tensors_info_get_size (std::addressof (inputTensorMeta), (gint) i);
    if (buf_size < (size_t) nns_size)
      throw std::runtime_error ("LiteRT input buffer " + std::to_string (i) + " is smaller ("
                                + std::to_string (buf_size) + " B) than the tensor ("
                                + std::to_string (nns_size) + " B).");
    input_tensor_sizes.push_back ((size_t) nns_size);
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

    size_t buf_size = 0;
    LITERT_CHECK (LiteRtGetTensorBufferSize (buf, &buf_size));
    gsize nns_size
        = gst_tensors_info_get_size (std::addressof (outputTensorMeta), (gint) i);
    if (buf_size < (size_t) nns_size)
      throw std::runtime_error ("LiteRT output buffer " + std::to_string (i) + " is smaller ("
                                + std::to_string (buf_size) + " B) than the tensor ("
                                + std::to_string (nns_size) + " B).");
    output_tensor_sizes.push_back ((size_t) nns_size);

    output_types.push_back (ranked_type);

    /** A larger LiteRT buffer means it carries padding the caller's tensor has
     *  no room for, so only an exact match may back the run directly. The size
     *  floor keeps the wrap off tensors whose copy is cheaper than it. */
    const bool wrappable = (buf_size == (size_t) nns_size)
                           && ((size_t) nns_size >= litert_wrap_min_bytes)
                           && requirementsAllowHostMemory (reqs);
    output_wrappable.push_back (wrappable ? 1 : 0);
  }

  invoke_outputs.resize (outputTensorMeta.num_tensors);
}

/**
 * @brief Whether a tensor's buffer requirements admit plain host memory.
 */
bool
litert_subplugin::requirementsAllowHostMemory (LiteRtTensorBufferRequirements reqs)
{
  int num_types = 0;

  if (LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes (reqs, &num_types) != kLiteRtStatusOk)
    return false;

  for (int i = 0; i < num_types; ++i) {
    LiteRtTensorBufferType type;

    if (LiteRtGetTensorBufferRequirementsSupportedTensorBufferType (reqs, i, &type) != kLiteRtStatusOk)
      return false;
    if (type == kLiteRtTensorBufferTypeHostMemory)
      return true;
  }

  return false;
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
    env = litert_env_ref ();

    /** Unwinding releases this before the handler below runs, so the
     *  cleanup() there can take the lock again without deadlocking. */
    std::lock_guard<std::shared_mutex> guard (litert_env_lock);

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

  /** Shared, so inference is never serialized across instances; see the
   *  shared-environment comment for why concurrent invoke is a reasoned
   *  choice rather than a documented guarantee. This only blocks while
   *  another instance is being configured or torn down. */
  std::shared_lock<std::shared_mutex> guard (litert_env_lock);

  /* Fill input buffers */
  for (i = 0; i < inputTensorMeta.num_tensors; ++i) {
    void *host_mem = nullptr;

    if (input[i].data == nullptr)
      throw std::invalid_argument ("Input tensor memory is null.");

    /** Reject any size mismatch: a larger input would overflow the LiteRT
     *  buffer, and a smaller one would leave the tail of the reused buffer
     *  holding the previous invoke's data (silently wrong inference). */
    if (input[i].size != input_tensor_sizes[i])
      throw std::invalid_argument ("Input tensor " + std::to_string (i) + " ("
                                   + std::to_string (input[i].size) + " B) does not match the model tensor ("
                                   + std::to_string (input_tensor_sizes[i]) + " B).");

    LITERT_CHECK (LiteRtLockTensorBuffer (
        input_buffers[i], &host_mem, kLiteRtTensorBufferLockModeWrite));
    std::memcpy (host_mem, input[i].data, input[i].size);
    LITERT_CHECK (LiteRtUnlockTensorBuffer (input_buffers[i]));
  }

  /** Let the run write straight into the caller's tensor where LiteRT will
   *  accept it, which drops the read-back copy below for that tensor.
   *
   *  Only outputs. The input side is mapped GST_MAP_READ by tensor_filter and
   *  its memory may be shared with another branch of the pipeline, so handing
   *  it out as a writable buffer would rest on LiteRT never writing to an
   *  input - which upstream does not state anywhere. */
  wrapped_buffers wrappers;

  for (i = 0; i < outputTensorMeta.num_tensors; ++i) {
    if (output[i].data == nullptr)
      throw std::invalid_argument ("Output tensor memory is null.");

    /** Reject any size mismatch: a larger output would read past the LiteRT
     *  buffer, and a smaller one would silently truncate the result. */
    if (output[i].size != output_tensor_sizes[i])
      throw std::invalid_argument ("Output tensor " + std::to_string (i) + " ("
                                   + std::to_string (output[i].size) + " B) does not match the model tensor ("
                                   + std::to_string (output_tensor_sizes[i]) + " B).");

    if (output_wrappable[i] && isHostMemoryAligned (output[i].data)) {
      LiteRtTensorBuffer buf = nullptr;

      LITERT_CHECK (LiteRtCreateTensorBufferFromHostMemory (
          &output_types[i], output[i].data, output[i].size, nullptr, &buf));
      wrappers.hold (buf);
      invoke_outputs[i] = buf;
    } else {
      invoke_outputs[i] = output_buffers[i];
    }
  }

  LITERT_CHECK (LiteRtRunCompiledModel (compiled_model, signature_index,
      input_buffers.size (), input_buffers.data (), invoke_outputs.size (),
      invoke_outputs.data ()));

  /* Read back the outputs that the run could not write into directly */
  for (i = 0; i < outputTensorMeta.num_tensors; ++i) {
    void *host_mem = nullptr;

    if (invoke_outputs[i] != output_buffers[i])
      continue;

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
