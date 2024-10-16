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

using namespace torch::executor;

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
  std::vector<TensorImpl> input_tensor_impl; /**< vector for input tensors */

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

  if (!configured)
    return;

  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));

  configured = false;
}

/**
 * @brief Method to prepare/configure ExecuTorch instance.
 */
void
executorch_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  try {
    /* Load network (.pte file) */
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

    /** @todo support more data types */
    auto convertType = [] (ScalarType type) {
      switch (type) {
        case ScalarType::Float:
          return _NNS_FLOAT32;
        default:
          return _NNS_END;
      }
    };

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

      /* get tensor dimension */
      auto sizes = input_meta->sizes ();
      const size_t rank = sizes.size ();
      for (size_t d = 0; d < rank; ++d) {
        const int dim = input_meta->sizes ()[d];
        info->dimension[rank - 1 - d] = (uint32_t) dim;
      }

      /* set tensor impl */
      input_tensor_impl.push_back (TensorImpl (type, rank,
          const_cast<TensorImpl::SizesType *> (sizes.data ()), nullptr));
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

      /* get tensor dimension */
      auto sizes = output_meta->sizes ();
      const size_t rank = sizes.size ();
      for (size_t d = 0; d < rank; ++d) {
        const int dim = output_meta->sizes ()[d];
        info->dimension[rank - 1 - d] = (uint32_t) dim;
      }
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
    TensorImpl &impl = input_tensor_impl[i];
    impl.set_data (input[i].data);
    input_values.push_back (Tensor (&impl));
  }

  const auto result = module->forward (input_values);
  ET_CHECK_MSG (result.ok (), "Failed to execute the model");

  /** @todo Remove memcpy of output tensor */
  for (size_t i = 0; i < outputInfo.num_tensors; ++i) {
    const auto result_tensor = result->at (i).toTensor ();
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
