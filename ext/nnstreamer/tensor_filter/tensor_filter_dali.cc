/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for dali
 * Copyright (C) 2024 Bram Veldhoen
 */
/**
 * @file        tensor_filter_dali.cc
 * @date        Jun 2024
 * @brief       NNStreamer tensor-filter sub-plugin for dali
 * @see         http://github.com/nnstreamer/nnstreamer
 * @see         https://github.com/NVIDIA/DALI
 * @author      Bram Veldhoen
 * @bug         No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (dali) for tensor_filter.
 *
 * This plugin uses a dali pipeline, which can be created and serialized with
 * c++. (see for instance tests/nnstreamer_filter_dali/main.cpp).
 * In order to support batch_size > 1, the dali pipeline always expects an
 * explicit batch size. Therefore, the dimensions specified for the dali
 * input/output should always include the batch size (even if it's 1). For
 * example, for batch_size 4:
 * @code
 * tensor_filter framework=dali model=dali_pipeline_4_3_640_640.txt
 * inputname=input0 input=3:2560:1440:4 inputtype=uint8 output=640:640:3:4
 * outputtype=float32
 * @endcode
 * This approach implies some specific requirements to the dali pipeline:
 * - The pipeline ctor parameter max_batch_size must be 1.
 * - The ExternalSource argument "batch" should be false.
 * - The ExternalSource argument "ndim" must match the number of dimensions
 * specified for the tensor_filter inputtype. For example:
 * @code
 * // Create the pipeline
 * Pipeline pipe(
 *   1, // max_batch_size
 *   ...
 * );
 *
 * // Add pipeline operators
 * // Input: (4, 1440, 2560, 3), Output: (4, 1440, 2560, 3)
 * pipe.AddOperator(
 *   OpSpec("ExternalSource")
 *   .AddArg("batch", false)
 *   .AddArg("device", "cpu")
 *   .AddArg("dtype", DALIDataType::DALI_UINT8)
 *   .AddArg("name", "input0")
 *   .AddArg("ndim", 4)
 *   .AddOutput("input0", "cpu"),
 *   "input0"
 * );
 * ... (add other operators) ...
 * // Build the pipeline
 * std::vector<std::pair<std::string, std::string>> outputs = {{"output0",
 * "gpu"}}; pipe.Build(outputs);
 *
 * // Serialize the pipeline
 * auto serialized_pipe_str = pipe.SerializeToProtobuf();
 *
 * // Save to file
 * std::ofstream serialized_pipe_out(serialized_pipe_filename);
 * serialized_pipe_out << serialized_pipe_str;
 * serialized_pipe_out.close();
 * @endcode
 *
 * @todo
 *  - Support multiple inputs/outputs.
 *  - Support for input/output devices ("cpu" or "gpu"). Currently only dali
 * input device = "cpu", outputdevice = "gpu" is supported.
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include <dali/c_api.h>
#include <dali/operators.h>
#include <dali/pipeline/data/types.h>


namespace
{

template <typename T>
inline std::string
to_string (const std::vector<T> &vec, const char sep = ',')
{
  std::stringstream ss;
  ss << "(";
  for (std::size_t index = 0; index < vec.size (); ++index) {
    const T &item = vec[index];
    if (index) {
      ss << sep << " ";
    }
    ss << item;
  }
  ss << ")";
  return ss.str ();
}

} // namespace
namespace nnstreamer
{
namespace tensor_filter_dali
{
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void _init_filter_dali (void) __attribute__ ((constructor));
void _fini_filter_dali (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/** @brief tensor-filter-subplugin concrete class for dali */
class dali_subplugin final : public tensor_filter_subplugin
{
  public:
  static void init_filter_dali ();
  static void fini_filter_dali ();

  dali_subplugin ();
  ~dali_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  private:
  bool configured{};
  int _device_id{}; /**< Device id of the gpu to use. */

  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw{};
  static dali_subplugin *registeredRepresentation;

  GstTensorsInfo _inputTensorMeta{};
  GstTensorsInfo _outputTensorMeta{};

  daliPipelineHandle _pipeline_handle{}; /**< The dali c_api handle to the pipeline. */

  // We currently use only the first input and first output of the dali pipeline.
  std::size_t _input_index{}; /** The index of the dali and nns input (i.e., 0) */
  std::size_t _output_index{}; /** The index of the dali and nns output (i.e., 0) */

  gchar *_pipeline_path{}; /**< pipeline file path */
  const char *_nns_input_name{}; /**< Name of the first input of the dali pipeline */
  std::vector<std::int64_t> _nns_input_shape{}; /**< The reversed shape of the nns input */
  dali_data_type_t _nns_input_dali_data_type{}; /**< The dali type of the nns input */
  const char *_nns_output_name{}; /**< Name of the first output of the dali pipeline */
  std::vector<std::int64_t> _nns_output_shape{}; /**< The reversed shape of the nns output */
  dali_data_type_t _nns_output_dali_data_type{}; /**< The dali type of the nns output */
  gsize _nns_output_tensor_element_size{}; /**< The element size of the dali output, i.e. the number of uint8's or float32's. */

  void cleanup ();
  void loadPipeline ();
  std::vector<std::int64_t> getDaliOutputShape ();
  std::vector<std::int64_t> convertShape (const GstTensorInfo &tensor_info) const;
  dali_data_type_t getDaliDataType (tensor_type nns_tensor_type) const;
};

const char *dali_subplugin::name = "dali";
const accl_hw dali_subplugin::hw_list[] = {};

/**
 * @brief Constructor for dali_subplugin.
 */
dali_subplugin::dali_subplugin ()
{
  ml_logi ("dali_subplugin::dali_subplugin");
  gst_tensors_info_init (&_inputTensorMeta);
  gst_tensors_info_init (&_outputTensorMeta);
}

/**
 * @brief Destructor for dali_subplugin.
 */
dali_subplugin::~dali_subplugin ()
{
  ml_logi ("dali_subplugin::~dali_subplugin");
  cleanup ();
}

/** @brief cleanup resources used by dali subplugin */
void
dali_subplugin::cleanup ()
{
  if (!configured) {
    return;
  }

  if (_pipeline_handle) {
    daliDeletePipeline (&_pipeline_handle);
    _pipeline_handle = nullptr;
  }

  gst_tensors_info_free (&_inputTensorMeta);
  gst_tensors_info_free (&_outputTensorMeta);

  configured = false;
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
dali_subplugin::getEmptyInstance ()
{
  ml_logi ("dali_subplugin::getEmptyInstance");
  return *(new dali_subplugin ());
}

/**
 * @brief Method to prepare/configure dali instance.
 */
void
dali_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  ml_logi ("dali_subplugin::configure_instance");
  g_assert (prop != nullptr);

  gst_tensors_info_copy (
      std::addressof (_inputTensorMeta), std::addressof (prop->input_meta));
  gst_tensors_info_copy (
      std::addressof (_outputTensorMeta), std::addressof (prop->output_meta));

  // Get dali input datatype
  GstTensorInfo *input_tensor_info = gst_tensors_info_get_nth_info (
      std::addressof (_inputTensorMeta), _input_index);
  _nns_input_name = input_tensor_info->name;
  _nns_input_dali_data_type = getDaliDataType (input_tensor_info->type);
  _nns_input_shape = convertShape (*input_tensor_info);

  // Get dali output datatype
  GstTensorInfo *output_tensor_info = gst_tensors_info_get_nth_info (
      std::addressof (_outputTensorMeta), _output_index);
  _nns_output_name = output_tensor_info->name;
  _nns_output_dali_data_type = getDaliDataType (output_tensor_info->type);
  _nns_output_tensor_element_size
      = gst_tensor_get_element_size (output_tensor_info->type);
  _nns_output_shape = convertShape (*output_tensor_info);

  ml_logi ("input_shape: %s, output shape: %s",
      to_string (_nns_input_shape).data (), to_string (_nns_output_shape).data ());

  /* Set pipeline path */
  if (prop->num_models != 1 || !prop->model_files[0]) {
    ml_loge ("dali filter requires one pipeline file.");
    throw std::invalid_argument ("The pipeline file is not given.");
  }
  _pipeline_path = g_strdup (prop->model_files[0]);
  g_assert (_pipeline_path != nullptr);

  loadPipeline ();
}

/**
 * @brief Loads the pipeline file and makes a member object to be used for execution.
 */
void
dali_subplugin::loadPipeline ()
{
  std::filesystem::path pipeline_fs_path (_pipeline_path);

  std::ifstream file (pipeline_fs_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg ();
  if (size < 0) {
    ml_loge ("Unable to open pipeline file %s", std::string (pipeline_fs_path).data ());
    throw std::runtime_error ("Unable to open pipeline file");
  }

  file.seekg (0, std::ios::beg);
  ml_logi ("Loading pipeline from file: %s with buffer size: %" G_GUINT64_FORMAT,
      std::string (pipeline_fs_path).data (), size);

  std::stringstream serialized_pipeline;
  serialized_pipeline << file.rdbuf ();
  auto serialized_pipeline_str = serialized_pipeline.str ();

  daliDeserializeDefault (&_pipeline_handle, serialized_pipeline_str.c_str (),
      serialized_pipeline_str.size ());

  configured = true;
}

/**
 * @brief Method to execute the dali pipeline.
 */
void
dali_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  try {
    ml_logd ("dali_subplugin::invoke on device_id: %d", _device_id);
    g_assert (configured);

    daliSetExternalInputBatchSize (&_pipeline_handle, _nns_input_name, 1);

    device_type_t device_type = CPU;
    auto data_ptr = input[_input_index].data;
    auto sample_dim = _nns_input_shape.size ();
    const char *layout_str = nullptr;
    unsigned int flags{ DALI_ext_default };
    ml_logd ("dali_subplugin::invoke; copy data");
    daliSetExternalInput (&_pipeline_handle, _nns_input_name, device_type,
        data_ptr, _nns_input_dali_data_type, _nns_input_shape.data (),
        sample_dim, layout_str, flags);
    ml_logd ("dali_subplugin::invoke; copied data");

    ml_logd ("dali_subplugin::invoke; execute pipeline");
    daliRun (&_pipeline_handle);
    ml_logd ("dali_subplugin::invoke; executed pipeline");

    // Collect outputs
    daliOutput (&_pipeline_handle);
    g_assert (daliGetNumOutput (&_pipeline_handle) == 1);

    bool has_uniform_shape = daliOutputHasUniformShape (&_pipeline_handle, _output_index);
    g_assert (has_uniform_shape);

    auto dali_output_data_type = daliTypeAt (&_pipeline_handle, _output_index);
    auto nns_output_num_elements = output[_output_index].size / _nns_output_tensor_element_size;
    auto dali_output_num_elements = daliNumElements (&_pipeline_handle, _output_index);
    auto dali_output_shape = getDaliOutputShape ();

    ml_logd ("nnstreamer output num_elements: %" G_GINT64_FORMAT
             "; dali output num_elements: %" G_GINT64_FORMAT "; dali output shape: %s",
        nns_output_num_elements, dali_output_num_elements,
        to_string (dali_output_shape).data ());

    g_assert (dali_output_data_type == _nns_output_dali_data_type);
    g_assert (nns_output_num_elements == dali_output_num_elements);
    g_assert (_nns_output_shape == dali_output_shape);

    ml_logd ("dali_subplugin::invoke; copying output");
    daliOutputCopy (&_pipeline_handle, output[_output_index].data,
        _output_index, device_type, 0, flags);
    ml_logd ("dali_subplugin::invoke; copied output");

  } catch (const std::exception &exc) {
    ml_loge ("ERROR running dali pipeline: %s", exc.what ());
    const std::string err_msg
        = "ERROR running dali pipeline: " + (std::string) exc.what ();
    throw std::runtime_error (err_msg);
  }
}

/**
 * @brief Method to get the information of dali subplugin.
 */
void
dali_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  ml_logi ("dali_subplugin::getFrameworkInfo");

  info.name = name;
  info.allow_in_place = FALSE;
  info.allocate_in_invoke = FALSE;
  info.run_without_model = FALSE;
  info.verify_model_path = TRUE;
  info.hw_list = hw_list;
  info.num_hw = num_hw;
}

/**
 * @brief Method to get the model information.
 */
int
dali_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  ml_logi ("dali_subplugin::getModelInfo");

  if (ops != GET_IN_OUT_INFO) {
    return -ENOENT;
  }

  gst_tensors_info_copy (std::addressof (in_info), std::addressof (_inputTensorMeta));
  gst_tensors_info_copy (std::addressof (out_info), std::addressof (_outputTensorMeta));

  return 0;
}

/**
 * @brief Method to handle events.
 */
int
dali_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  ml_logi ("dali_subplugin::eventHandler");

  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

std::vector<std::int64_t>
dali_subplugin::convertShape (const GstTensorInfo &tensor_info) const
{
  std::vector<std::int64_t> shape;

  // Set dali shape in reverse order
  int dim_index = 0;
  while (tensor_info.dimension[dim_index] != 0) {
    shape.push_back (tensor_info.dimension[dim_index++]);
  }
  std::reverse (shape.begin (), shape.end ());

  return shape;
}

std::vector<std::int64_t>
dali_subplugin::getDaliOutputShape ()
{
  std::vector<std::int64_t> dali_output_shape;

  auto custom_deleter = [] (std::int64_t *t) noexcept { free (t); };
  auto dali_output_shape_ptr = std::unique_ptr<int64_t, decltype (custom_deleter)> (
      daliShapeAt (&_pipeline_handle, _output_index), custom_deleter);

  // The dali output has an additional dimension with value 1 at index 0, i.e. (1, 4, 3, 640, 640) instead of (4, 3, 640, 640).
  //  We can safely ignore this additional dimension and remove it. Therefore start index at 1.
  std::size_t shape_index = 1;
  while (dali_output_shape_ptr.get ()[shape_index] != 0) {
    dali_output_shape.push_back (dali_output_shape_ptr.get ()[shape_index++]);
  }

  return dali_output_shape;
}

dali_data_type_t
dali_subplugin::getDaliDataType (tensor_type nns_tensor_type) const
{
  switch (nns_tensor_type) {
    case _NNS_INT32:
      return DALI_INT32;
    case _NNS_UINT32:
      return DALI_UINT32;
    case _NNS_INT16:
      return DALI_INT16;
    case _NNS_UINT16:
      return DALI_UINT16;
    case _NNS_INT8:
      return DALI_INT8;
    case _NNS_UINT8:
      return DALI_UINT8;
    case _NNS_FLOAT64:
      return DALI_FLOAT64;
    case _NNS_FLOAT32:
      return DALI_FLOAT;
    case _NNS_INT64:
      return DALI_INT64;
    case _NNS_UINT64:
      return DALI_UINT64;
    case _NNS_FLOAT16:
      return DALI_FLOAT16;
    case _NNS_END:
    default:
      throw std::runtime_error ("tensor_type not supported.");
  }
}

dali_subplugin *dali_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register. */
void
dali_subplugin::init_filter_dali ()
{
  ml_logi ("dali_subplugin::init_filter_dali");

  dali::InitOperatorsLib ();
  daliInitialize ();

  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<dali_subplugin> ();
}

/** @brief Destruct the sub-plugin for dali. */
void
dali_subplugin::fini_filter_dali ()
{
  ml_logi ("dali_subplugin::fini_filter_dali");

  g_assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/** @brief initializer */
void
_init_filter_dali ()
{
  dali_subplugin::init_filter_dali ();
}

/** @brief finalizer */
void
_fini_filter_dali ()
{
  dali_subplugin::fini_filter_dali ();
}

} /* namespace tensor_filter_dali */
} /* namespace nnstreamer */
