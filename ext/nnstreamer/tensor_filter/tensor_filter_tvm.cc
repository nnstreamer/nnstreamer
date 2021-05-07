/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    tensor_filter_tvm.cc
 * @date    16 Apr 2021
 * @brief   NNStreamer tensor-filter sub-plugin for Apache TVM
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * This is the per-NN-framework plugin (TVM) for tensor_filter.
 *
 * @note    Only supports tvm.contrib.graph_executor model
 */

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <tensor_common.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>


namespace nnstreamer
{
namespace tensorfilter_tvm
{

G_BEGIN_DECLS

void init_filter_tvm (void) __attribute__ ((constructor));
void fini_filter_tvm (void) __attribute__ ((destructor));

G_END_DECLS

/**
 * @brief Class for TVM subplugin.
 */
class tvm_subplugin final : public tensor_filter_subplugin
{
  private:
  bool empty_model;
  gchar *model_path;
  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;

  DLDevice device;
  tvm::runtime::Module mod_factory;
  tvm::runtime::Module gmod;
  std::vector<DLTensor> input_tensor_list;
  std::vector<DLTensor> output_tensor_list;
  bool zero_copy_enabled;

  static const char *name;
  static const accl_hw hw_list[];
  static tvm_subplugin *registeredRepresentation;

  bool parse_custom_prop (const char *custom_prop);
  bool convert_dtype (tensor_type &nns_type, const DLDataType &dtype);
  void cleanup () noexcept;

  public:
  static void init_filter_tvm ();
  static void fini_filter_tvm ();

  tvm_subplugin ();
  ~tvm_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *tvm_subplugin::name = "tvm";
const accl_hw tvm_subplugin::hw_list[] = { ACCL_CPU, ACCL_GPU };

/**
 * @brief Construct a new tvm subplugin::tvm subplugin object
 */
tvm_subplugin::tvm_subplugin ()
    : tensor_filter_subplugin (), empty_model (true), model_path (nullptr),
      device (DLDevice{ kDLCPU, 0 }), mod_factory (nullptr), gmod (nullptr),
      zero_copy_enabled (true)
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));
}

/**
 * @brief Cleanup method for tvm subplugin
 */
void
tvm_subplugin::cleanup () noexcept
{
  if (empty_model)
    return;
  g_free (model_path);

  input_tensor_list.clear ();
  output_tensor_list.clear ();
  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));

  model_path = nullptr;
  empty_model = true;
}

/**
 * @brief Destroy the tvm subplugin::tvm subplugin object
 */
tvm_subplugin::~tvm_subplugin ()
{
  cleanup ();
}

/**
 * @brief Method to get an empty object
 */
tensor_filter_subplugin &
tvm_subplugin::getEmptyInstance ()
{
  return *(new tvm_subplugin ());
}

/**
 * @brief Internal method to parse custom properties
 * @param custom_prop Given c_str value of 'custom' property,
 *                    which contains device info
 */
bool
tvm_subplugin::parse_custom_prop (const char *custom_prop)
{
  gchar **options;
  bool invalid_option = false;

  if (custom_prop == nullptr) {
    /* no custom properties */
    return true;
  }

  options = g_strsplit (custom_prop, ",", -1);

  for (guint op = 0; op < g_strv_length (options); ++op) {
    gchar **option = g_strsplit (options[op], ":", -1);

    if (g_strv_length (option) > 1) {
      g_strstrip (option[0]);
      g_strstrip (option[1]);

      if (g_ascii_strcasecmp (option[0], "device") == 0) {
        if (g_ascii_strcasecmp (option[1], "CPU") == 0) {
          device = DLDevice{ kDLCPU, 0 };
        } else if (g_ascii_strcasecmp (option[1], "GPU") == 0) {
          device = DLDevice{ kDLGPU, 0 };
        } else {
          nns_loge ("Unknown device (%s).", option[1]);
          invalid_option = true;
        }
      } else {
        nns_logw ("Unknown option (%s).", options[op]);
      }
    }
    if (invalid_option) {
      g_strfreev (option);
      break;
    }
    g_strfreev (option);
  }
  g_strfreev (options);
  return !invalid_option;
}

/**
 * @brief Configure tvm instance
 */
void
tvm_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  if (!parse_custom_prop (prop->custom_properties)) {
    nns_loge ("Failed to parse custom property.");
    cleanup ();
    throw std::invalid_argument ("Failed to parse custom property.");
  }

  if (!empty_model) {
    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      std::cerr << "Model path is not given." << std::endl;
      cleanup ();
      throw std::invalid_argument ("Model path is not given.");
    }
    cleanup ();
  }

  /* read model */
  model_path = g_strdup (prop->model_files[0]);
  mod_factory = tvm::runtime::Module::LoadFromFile (model_path, "so");
  gmod = mod_factory.GetFunction ("default") (device);

  inputInfo.num_tensors = (int) gmod.GetFunction ("get_num_inputs") ();
  outputInfo.num_tensors = (int) gmod.GetFunction ("get_num_outputs") ();

  tvm::runtime::NDArray arr;
  const DLTensor *dt;

  /* infer in out info */
  auto getInput = gmod.GetFunction ("get_input");
  if (getInput == nullptr) {
    cleanup ();
    throw std::invalid_argument ("Packed function `get_input` not defined in model");
  }
  auto getOutput = gmod.GetFunction ("get_output");
  if (getOutput == nullptr) {
    cleanup ();
    throw std::invalid_argument ("Packed function `get_output` not defined in model");
  }

  for (unsigned int i = 0; i < inputInfo.num_tensors; ++i) {
    arr = getInput (i);
    dt = arr.operator-> ();
    input_tensor_list.push_back (*dt);

    if (!convert_dtype (inputInfo.info[i].type, dt->dtype)) {
      cleanup ();
      throw std::invalid_argument ("Failed to convert DLPack data type");
    }
    std::copy (dt->shape, dt->shape + dt->ndim, inputInfo.info[i].dimension);
    inputInfo.info[i].name = nullptr;
  }

  for (unsigned int i = 0; i < outputInfo.num_tensors; ++i) {
    arr = getOutput (i);
    dt = arr.operator-> ();
    output_tensor_list.push_back (*dt);

    if (!convert_dtype (outputInfo.info[i].type, dt->dtype)) {
      cleanup ();
      throw std::invalid_argument ("Failed to convert DLPack data type");
    }
    std::copy (dt->shape, dt->shape + dt->ndim, outputInfo.info[i].dimension);
    outputInfo.info[i].name = nullptr;
  }
  empty_model = false;
}

/**
 * @brief Internal method to convert DLPack data type to tensor type
 */
bool
tvm_subplugin::convert_dtype (tensor_type &nns_type, const DLDataType &dtype)
{
  switch (dtype.code) {
    case kDLInt:
      switch (dtype.bits) {
        case 64:
          nns_type = _NNS_INT64;
          break;
        case 32:
          nns_type = _NNS_INT32;
          break;
        case 16:
          nns_type = _NNS_INT16;
          break;
        case 8:
          nns_type = _NNS_INT8;
          break;
        default:
          return false;
      }
      break;
    case kDLUInt:
      switch (dtype.bits) {
        case 64:
          nns_type = _NNS_UINT64;
          break;
        case 32:
          nns_type = _NNS_UINT32;
          break;
        case 16:
          nns_type = _NNS_UINT16;
          break;
        case 8:
          nns_type = _NNS_UINT8;
          break;
        default:
          return false;
      }
      break;
    case kDLFloat:
      switch (dtype.bits) {
        case 64:
          nns_type = _NNS_FLOAT64;
          break;
        case 32:
          nns_type = _NNS_FLOAT32;
          break;
        default:
          return false;
      }
      break;
    default:
      return false;
  }
  return true;
}

/**
 * @brief Invoke tvm instance
 */
void
tvm_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  assert (!empty_model);
  assert (gmod.defined ());
  assert (input != NULL && output != NULL);

  unsigned int i;
  tvm::runtime::NDArray tensor;
  tvm::runtime::PackedFunc set_input, set_input_zero_copy, get_output, run;

  /* input data is aligned */
  set_input_zero_copy = gmod.GetFunction ("set_input_zero_copy");
  if (set_input_zero_copy == nullptr) {
    cleanup ();
    throw std::runtime_error ("Packed function `set_input_zero_copy` not defined in model");
  }
  set_input = gmod.GetFunction ("set_input");
  if (set_input == nullptr) {
    cleanup ();
    throw std::runtime_error ("Packed function `set_input` not defined in model");
  }
  get_output = gmod.GetFunction ("get_output");
  if (get_output == nullptr) {
    cleanup ();
    throw std::runtime_error ("Packed function `get_output` not defined in model");
  }
  run = gmod.GetFunction ("run");
  if (run == nullptr) {
    cleanup ();
    throw std::runtime_error ("Packed function `run` not defined in model");
  }

  for (i = 0; i < inputInfo.num_tensors; ++i) {
    input_tensor_list[i].data = input[i].data;

    if (zero_copy_enabled) {
      try {
        set_input_zero_copy (i, &input_tensor_list[i]);
      } catch (const std::runtime_error &e) {
        nns_logw ("Input data is not aligned, which results in memory copy.");
        zero_copy_enabled = false;
        set_input (i, &input_tensor_list[i]);
      }
    } else {
      set_input (i, &input_tensor_list[i]);
    }
  }

  run ();

  for (i = 0; i < outputInfo.num_tensors; ++i) {
    output_tensor_list[i].data = output[i].data;
    get_output (i, &output_tensor_list[i]);
  }
}

/**
 * @brief Get tvm frameworks info
 */
void
tvm_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
}

/**
 * @brief Get tvm model information
 */
int
tvm_subplugin::getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle the event
 */
int
tvm_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  return -ENOENT;
}

tvm_subplugin *tvm_subplugin::registeredRepresentation = nullptr;

/**
 * @brief Initialize the object for runtime register
 */
void
tvm_subplugin::init_filter_tvm (void)
{
  /* mem alignment */
  gst_tensor_alloc_init (127);
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<tvm_subplugin> ();
}

/**
 * @brief Destruct the subplugin
 */
void
tvm_subplugin::fini_filter_tvm (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief initializer
 */
void
init_filter_tvm ()
{
  tvm_subplugin::init_filter_tvm ();
}

/**
 * @brief finalizer
 */
void
fini_filter_tvm ()
{
  tvm_subplugin::fini_filter_tvm ();
}

} // namespace tensorfilter_tvm
} /* namespace nnstreamer */
