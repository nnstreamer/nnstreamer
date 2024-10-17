/* SPDX-License-Identifier: LGPL-2.1-only */

/**
 * @file    tensor_filter_executorch_llama.cc
 * @date    16 Oct 2024
 * @brief   NNStreamer tensor-filter sub-plugin for ExecuTorch Llama
 * @author
 * @see     https://github.com/nnstreamer/nnstreamer
 *          https://github.com/pytorch/executorch/blob/main/examples/models/llama
 * @bug     No known bugs.
 * @note    Currently experimental.
 * @details A pipeline example:
 *          gst-launch-1.0  filesrc location=input.txt ! application/octet-stream !
 *          tensor_converter ! other/tensors,format=flexible !
 *          tensor_filter framework=executorch-llama model=model.pte,tokenizer.model invoke-dynamic=true !
 *          other/tensors,format=flexible ! tensor_decoder mode=octet_stream !
 *          application/octet-stream ! filesink location=result.txt
 */

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <executorch/examples/models/llama/runner/runner.h>

using namespace torch::executor;

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
  std::unique_ptr<example::Runner> runner; /**< Runner for Llama model */

  public:
  static void init_filter_executorch_llama ();
  static void fini_filter_executorch_llama ();

  executorch_llama_subplugin (){};
  ~executorch_llama_subplugin (){};

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void invoke_dynamic (GstTensorFilterProperties *prop,
      const GstTensorMemory *input, GstTensorMemory *output) override;
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *executorch_llama_subplugin::fw_name = "executorch-llama";

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
executorch_llama_subplugin::getEmptyInstance ()
{
  return *(new executorch_llama_subplugin ());
}

/**
 * @brief Method to prepare/configure ExecuTorch instance.
 */
void
executorch_llama_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  try {
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

    /* set llama runner */
    runner = std::make_unique<example::Runner> (
        prop->model_files[0], prop->model_files[1], 0.8f);
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

  std::string prompt ((char *) input[0].data, input[0].size);
  std::string result = "";

  ET_CHECK_MSG (
      runner->generate (prompt, 128,
          [&result] (const std::string &generated) { result.append (generated); })
          == Error::Ok,
      "Generating text failed");

  /* replace null character with blankspace */
  std::replace (result.begin (), result.end (), '\0', ' ');

  output[0].size = result.size ();
  output[0].data = g_malloc0 (output[0].size);
  memcpy (output[0].data, result.c_str (), result.size ());

  gst_tensors_info_init (&prop->output_meta);
  prop->output_meta.num_tensors = 1;
  prop->output_meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  prop->output_meta.info[0].type = _NNS_UINT8;
  prop->output_meta.info[0].dimension[0] = result.size ();
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
