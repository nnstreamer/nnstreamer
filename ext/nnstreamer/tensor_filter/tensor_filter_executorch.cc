/* SPDX-License-Identifier: LGPL-2.1-only */

/**
 * @file    tensor_filter_executorch.cc
 * @date    26 Apr 2024
 * @brief   NNStreamer tensor-filter sub-plugin for ExecuTorch
 * @author
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs.
 *
 * This is the executorch plugin for tensor_filter.
 *
 * @note Currently only skeleton
 */

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::TensorShapeDynamism;
using executorch::extension::make_tensor_ptr;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace nnstreamer
{
namespace tensorfilter_executorch
{

G_BEGIN_DECLS

void init_filter_executorch (void) __attribute__ ((constructor));
void fini_filter_executorch (void) __attribute__ ((destructor));

G_END_DECLS

/**
 * @brief tensor-filter-subplugin concrete class for ExecuTorch
 */
class executorch_subplugin final : public tensor_filter_subplugin
{
  private:
  static executorch_subplugin *registeredRepresentation;
  static const GstTensorFilterFrameworkInfo framework_info;

  bool configured;
  char *model_path; /**< The model *.pte file */
  void cleanup (); /**< cleanup function */
  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */

  /* executorch */
  std::unique_ptr<Module> module; /**< model module */
  std::vector<TensorPtr> input_tensors; /**< input tensors, re-pointed at each invoke */
  std::vector<TensorPtr> output_tensors; /**< outputs ExecuTorch writes into the caller's buffers; null if memory-planned */
  std::vector<size_t> output_nbytes; /**< byte size of each output, from the method meta */

  static tensor_type convertType (ScalarType type);

  public:
  static void init_filter_executorch ();
  static void fini_filter_executorch ();

  executorch_subplugin ();
  ~executorch_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

/**
 * @brief Describe framework information.
 */
const GstTensorFilterFrameworkInfo executorch_subplugin::framework_info = { .name = "executorch",
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .verify_model_path = TRUE,
  .hw_list = (const accl_hw[]){ ACCL_CPU },
  .num_hw = 1,
  .accl_auto = ACCL_CPU,
  .accl_default = ACCL_CPU,
  .statistics = nullptr };

/**
 * @brief Constructor for executorch subplugin.
 */
executorch_subplugin::executorch_subplugin ()
    : tensor_filter_subplugin (), configured (false), model_path (nullptr)
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));
}

/**
 * @brief Destructor for executorch subplugin.
 */
executorch_subplugin::~executorch_subplugin ()
{
  cleanup ();
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
executorch_subplugin::getEmptyInstance ()
{
  return *(new executorch_subplugin ());
}

/**
 * @brief Method to cleanup executorch subplugin.
 */
void
executorch_subplugin::cleanup ()
{
  g_free (model_path);
  model_path = nullptr;

  /* A refused model may have filled part of the infos before configured is set. */
  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));
  input_tensors.clear ();
  output_tensors.clear ();
  output_nbytes.clear ();

  configured = false;
}

/**
 * @brief Convert an ExecuTorch scalar type to the NNStreamer tensor type.
 * @return _NNS_END if NNStreamer has no type with the same memory layout.
 */
tensor_type
executorch_subplugin::convertType (ScalarType type)
{
  switch (type) {
    case ScalarType::Byte:
    case ScalarType::Bool:
      return _NNS_UINT8;
    case ScalarType::Char:
      return _NNS_INT8;
    case ScalarType::Short:
      return _NNS_INT16;
    case ScalarType::Int:
      return _NNS_INT32;
    case ScalarType::Long:
      return _NNS_INT64;
    case ScalarType::Float:
      return _NNS_FLOAT32;
    case ScalarType::Double:
      return _NNS_FLOAT64;
    case ScalarType::Half:
#ifdef FLOAT16_SUPPORT
      return _NNS_FLOAT16;
#else
      ml_loge ("NNStreamer requires -DFLOAT16_SUPPORT as a build option to enable float16 type. This binary does not have float16 feature enabled; thus, float16 type is not supported in this instance.");
      break;
#endif
    default:
      break;
  }

  return _NNS_END;
}

/**
 * @brief Method to prepare/configure ExecuTorch instance.
 */
void
executorch_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  try {
    /* Load network (.pte file) */
    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      throw std::invalid_argument ("Model path is not given.");
    }

    if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
      const std::string err_msg
          = "Given file " + (std::string) prop->model_files[0] + " is not valid";
      throw std::invalid_argument (err_msg);
    }

    model_path = g_strdup (prop->model_files[0]);

    module = std::make_unique<Module> (model_path);
    if (module->load () != Error::Ok) {
      const std::string err_msg
          = "Failed to load module with Given file " + (std::string) model_path;
      throw std::invalid_argument (err_msg);
    }

    ET_CHECK_MSG (module->is_loaded (), "Making module failed");

    const auto forward_method_meta = module->method_meta ("forward");
    ET_CHECK_MSG (forward_method_meta.ok (), "Getting method meta failed");

    /* parse input tensors info */
    size_t num_inputs = forward_method_meta->num_inputs ();
    inputInfo.num_tensors = num_inputs;
    for (size_t i = 0; i < num_inputs; ++i) {
      const auto input_meta = forward_method_meta->input_tensor_meta (i);
      GstTensorInfo *info
          = gst_tensors_info_get_nth_info (std::addressof (inputInfo), i);

      /* get tensor data type */
      ScalarType type = input_meta->scalar_type ();
      info->type = convertType (type);
      if (info->type == _NNS_END)
        throw std::invalid_argument ("Unsupported data type of input tensor "
                                     + std::to_string (i) + ": ScalarType "
                                     + std::to_string ((int) type));

      /* get tensor dimension */
      auto sizes = input_meta->sizes ();
      const size_t rank = sizes.size ();
      for (size_t d = 0; d < rank; ++d) {
        const int dim = input_meta->sizes ()[d];
        info->dimension[rank - 1 - d] = (uint32_t) dim;
      }

      input_tensors.push_back (
          make_tensor_ptr (std::vector<SizesType> (sizes.begin (), sizes.end ()),
              static_cast<void *> (nullptr), type, TensorShapeDynamism::STATIC));
    }

    /* parse output tensors info */
    size_t num_outputs = forward_method_meta->num_outputs ();
    outputInfo.num_tensors = num_outputs;
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto output_meta = forward_method_meta->output_tensor_meta (i);
      GstTensorInfo *info
          = gst_tensors_info_get_nth_info (std::addressof (outputInfo), i);

      /* get tensor data type */
      ScalarType type = output_meta->scalar_type ();
      info->type = convertType (type);
      if (info->type == _NNS_END)
        throw std::invalid_argument ("Unsupported data type of output tensor "
                                     + std::to_string (i) + ": ScalarType "
                                     + std::to_string ((int) type));

      /* get tensor dimension */
      auto sizes = output_meta->sizes ();
      const size_t rank = sizes.size ();
      for (size_t d = 0; d < rank; ++d) {
        const int dim = output_meta->sizes ()[d];
        info->dimension[rank - 1 - d] = (uint32_t) dim;
      }

      output_nbytes.push_back (output_meta->nbytes ());

      /* A memory-planned output lives in the method's arena and cannot be redirected. */
      if (output_meta->is_memory_planned ())
        output_tensors.push_back (nullptr);
      else
        output_tensors.push_back (
            make_tensor_ptr (std::vector<SizesType> (sizes.begin (), sizes.end ()),
                static_cast<void *> (nullptr), type, TensorShapeDynamism::STATIC));
    }

    configured = true;
  } catch (const std::exception &e) {
    cleanup ();
    /* throw exception upward */
    throw;
  }
}

/**
 * @brief Method to execute the model.
 */
void
executorch_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  std::vector<EValue> input_values;
  for (size_t i = 0; i < inputInfo.num_tensors; ++i) {
    input_tensors[i]->unsafeGetTensorImpl ()->set_data (input[i].data);
    input_values.push_back (*input_tensors[i]);
  }

  for (size_t i = 0; i < outputInfo.num_tensors; ++i) {
    if (output[i].size < output_nbytes[i])
      throw std::runtime_error ("Output buffer " + std::to_string (i) + " is too small.");
  }

  /* The caller passes new output buffers on every invoke, so they are re-pointed each time. */
  for (size_t i = 0; i < outputInfo.num_tensors; ++i) {
    TensorPtr &tensor = output_tensors[i];

    if (!tensor)
      continue;

    tensor->unsafeGetTensorImpl ()->set_data (output[i].data);
    if (module->set_output (*tensor, i) != Error::Ok)
      throw std::runtime_error ("Failed to set output buffer " + std::to_string (i));
  }

  const auto result = module->forward (input_values);
  ET_CHECK_MSG (result.ok (), "Failed to execute the model");

  for (size_t i = 0; i < outputInfo.num_tensors; ++i) {
    const auto result_tensor = result->at (i).toTensor ();

    if (result_tensor.const_data_ptr () == output[i].data)
      continue;

    std::memcpy (output[i].data, result_tensor.const_data_ptr (), result_tensor.nbytes ());
  }
}

/**
 * @brief Method to get the information of ExecuTorch subplugin.
 */
void
executorch_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info = executorch_subplugin::framework_info;
}

/**
 * @brief Method to get the model information.
 */
int
executorch_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle events.
 */
int
executorch_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);

  return -ENOENT;
}

executorch_subplugin *executorch_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
executorch_subplugin::init_filter_executorch (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<executorch_subplugin> ();
}

/** @brief Destruct the subplugin */
void
executorch_subplugin::fini_filter_executorch (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief Register the sub-plugin for ExecuTorch.
 */
void
init_filter_executorch ()
{
  executorch_subplugin::init_filter_executorch ();
}

/**
 * @brief Destruct the sub-plugin for ExecuTorch.
 */
void
fini_filter_executorch ()
{
  executorch_subplugin::fini_filter_executorch ();
}

} /* namespace tensorfilter_executorch */
} /* namespace nnstreamer */
