/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter, TensorRT Module
 * Copyright (C) 2024 Bram Veldhoen
 */
/**
 * @file   tensor_filter_tensorrt10.cc
 * @date   Jun 2024
 * @brief  TensorRT 10+ module for tensor_filter gstreamer plugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @see    https://github.com/NVIDIA/TensorRT
 * @author Bram Veldhoen
 * @bug    No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (TensorRT) for tensor_filter.
 *
 * @note Supports onnxruntime .onnx and tensorrt .engine file as inference model formats.
 *   When an .onnx file is provided, this plugin will generate the tensorrt .engine file,
 *   and store it in /tmp/<modelfilename>.engine.
 *
 * @todo:
 *  - Add option parameter for generated .engine file (now default in /tmp).
 *  - Add support for model builder parameters.
 *  - Add support for optimization profile(s).
 *  - Add support for batch_size > 1 (to allow for i.e. multiple camera streams).
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

using Severity = nvinfer1::ILogger::Severity;

/** @brief a global object of ILogger */
class Logger : public nvinfer1::ILogger
{
  void log (Severity severity, const char *msg) noexcept override
  {
    switch (severity) {
      case Severity::kWARNING:
        ml_logw ("NVINFER: %s", msg);
        break;
      case Severity::kINFO:
        ml_logi ("NVINFER: %s", msg);
        break;
      case Severity::kVERBOSE:
        ml_logd ("NVINFER: %s", msg);
        break;
      default:
        ml_loge ("NVINFER: %s", msg);
        break;
    }
  }
} gLogger;

using nnstreamer::tensor_filter_subplugin;

namespace nnstreamer
{
namespace tensorfilter_tensorrt10
{
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void _init_filter_tensorrt10 (void) __attribute__ ((constructor));
void _fini_filter_tensorrt10 (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/** @brief Deleter for instances from the TensorRT library. */
struct NvInferDeleter {
  template <typename T> void operator() (T *obj) const
  {
    delete obj;
  }
};

template <typename T>
using NvInferUniquePtr = std::unique_ptr<T, NvInferDeleter>;

template <typename T>
NvInferUniquePtr<T>
makeNvInferUniquePtr (T *t)
{
  return NvInferUniquePtr<T>{ t };
}

/** @brief Holds metadata related to a TensorRT tensor. */
struct NvInferTensorInfo {
  const char *name;
  nvinfer1::TensorIOMode mode;
  nvinfer1::Dims shape;
  nvinfer1::DataType dtype;
  std::size_t dtype_size;
  std::size_t volume;
  std::size_t buffer_size;
  void *buffer; /**< Cuda buffer */
};

/** @brief tensorrt10 subplugin class */
class tensorrt10_subplugin final : public tensor_filter_subplugin
{
  template <typename T> using UniquePtr = std::unique_ptr<T>;

  public:
  static void init_filter_tensorrt10 ();
  static void fini_filter_tensorrt10 ();

  tensorrt10_subplugin ();
  ~tensorrt10_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);


  private:
  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw = 0;
  static tensorrt10_subplugin *registeredRepresentation;

  bool _configured{}; /**< Flag to keep track of whether this instance has been configured or not. */
  int _device_id{}; /**< Device id of the gpu to use. */
  gchar *_model_path{}; /**< engine file path */
  std::filesystem::path _engine_fs_path{}; /**< filesystem path to engine file */
  cudaStream_t _stream{}; /**< The cuda inference stream */

  GstTensorsInfo _inputTensorMeta;
  GstTensorsInfo _outputTensorMeta;

  NvInferUniquePtr<nvinfer1::IRuntime> _Runtime{};
  NvInferUniquePtr<nvinfer1::ICudaEngine> _Engine{};
  NvInferUniquePtr<nvinfer1::IExecutionContext> _Context{};

  std::vector<NvInferTensorInfo> _tensorrt10_input_tensor_infos{};
  std::vector<NvInferTensorInfo> _tensorrt10_output_tensor_infos{};

  void cleanup ();
  void allocBuffer (void **buffer, gsize size);
  void loadModel (const GstTensorFilterProperties *prop);
  void checkUnifiedMemory () const;
  void convertTensorsInfo (const std::vector<NvInferTensorInfo> &tensorrt10_tensor_infos,
      GstTensorsInfo &info) const;
  std::size_t getVolume (const nvinfer1::Dims &shape) const;
  std::size_t getTensorRTDataTypeSize (nvinfer1::DataType tensorrt10_data_type) const;
  tensor_type getNnStreamerDataType (nvinfer1::DataType tensorrt10_data_type) const;

  void constructNetwork (NvInferUniquePtr<nvonnxparser::IParser> &parser) const;
  void buildSaveEngine () const;
  void loadEngine ();
  void parseCustomProperties (const GstTensorFilterProperties *prop);
};

const char *tensorrt10_subplugin::name = "tensorrt10";
const accl_hw tensorrt10_subplugin::hw_list[] = {};

/**
 * @brief constructor of tensorrt10_subplugin
 */
tensorrt10_subplugin::tensorrt10_subplugin () : tensor_filter_subplugin ()
{
}

/**
 * @brief destructor of tensorrt10_subplugin
 */
tensorrt10_subplugin::~tensorrt10_subplugin ()
{
  cleanup ();
}

void
tensorrt10_subplugin::cleanup ()
{
  if (!_configured) {
    return;
  }

  gst_tensors_info_free (&_inputTensorMeta);
  gst_tensors_info_free (&_outputTensorMeta);

  for (auto &tensorrt10_tensor_info : _tensorrt10_input_tensor_infos) {
    cudaFree (tensorrt10_tensor_info.buffer);
    tensorrt10_tensor_info.buffer = nullptr;
  }

  if (_model_path != nullptr) {
    g_free (_model_path);
    _model_path = nullptr;
  }

  if (_stream) {
    cudaStreamDestroy (_stream);
  }
}

/**
 * @brief Returns an empty instance.
 * @return an empty instance
 */
tensor_filter_subplugin &
tensorrt10_subplugin::getEmptyInstance ()
{
  return *(new tensorrt10_subplugin ());
}

/**
 * @brief Parse custom properties and set instance options accordingly.
 */
void
tensorrt10_subplugin::parseCustomProperties (const GstTensorFilterProperties *prop)
{
  if (!prop->custom_properties) {
    ml_logd ("No custom properties provided");
    return;
  }
  using uniq_g_strv = std::unique_ptr<gchar *, std::function<void (gchar **)>>;
  guint len, i;

  uniq_g_strv options (g_strsplit (prop->custom_properties, ",", -1), g_strfreev);
  len = g_strv_length (options.get ());

  for (i = 0; i < len; ++i) {
    uniq_g_strv option (g_strsplit (options.get ()[i], ":", -1), g_strfreev);

    if (g_strv_length (option.get ()) > 1) {
      g_strstrip (option.get ()[0]);
      g_strstrip (option.get ()[1]);

      if (g_ascii_strcasecmp (option.get ()[0], "DeviceId") == 0) {
        _device_id = static_cast<int> (g_ascii_strtoull (option.get ()[1], NULL, 10));
      } else {
        ml_loge ("Unknown custom property: %s", option.get ()[0]);
        throw std::invalid_argument (
            "Unknown custom property: " + std::string (option.get ()[0]));
      }
    }
  }
}

/**
 * @brief Configure the instance of the tensorrt10_subplugin.
 * @param[in] prop property of tensor_filter instance
 */
void
tensorrt10_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  /* Set model path */
  if (prop->num_models != 1 || !prop->model_files[0]) {
    ml_loge ("TensorRT filter requires one engine model file.");
    throw std::invalid_argument ("The .engine model file is not given.");
  }
  assert (_model_path == nullptr);
  _model_path = g_strdup (prop->model_files[0]);

  try {
    parseCustomProperties (prop);
  } catch (const std::invalid_argument &e) {
    ml_loge ("Failed to parse custom property: %s", e.what ());
    throw std::invalid_argument (
        "Failed to parse custom property: " + std::string (e.what ()));
  }

  /* Make a TensorRT engine */
  loadModel (prop);

  _configured = true;
}

/**
 * @brief Invoke the TensorRT model and get the inference result.
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 */
void
tensorrt10_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  ml_logi ("tensorrt10_subplugin::invoke");
  g_assert (_configured);

  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  cudaError_t status;

  /* Copy input data to Cuda memory space */
  for (std::size_t i = 0; i < _tensorrt10_input_tensor_infos.size (); ++i) {
    const auto &tensorrt10_tensor_info = _tensorrt10_input_tensor_infos[i];
    g_assert (tensorrt10_tensor_info.buffer_size == input[i].size);

    status = cudaMemcpyAsync (tensorrt10_tensor_info.buffer, input[i].data,
        input[i].size, cudaMemcpyHostToDevice, _stream);

    if (status != cudaSuccess) {
      ml_loge ("Failed to copy to cuda input buffer");
      throw std::runtime_error ("Failed to copy to cuda input buffer");
    }
  }

  for (std::size_t i = 0; i < _tensorrt10_output_tensor_infos.size (); ++i) {
    const auto &tensorrt10_tensor_info = _tensorrt10_output_tensor_infos[i];
    g_assert (tensorrt10_tensor_info.buffer_size == output[i].size);
    allocBuffer (&output[i].data, output[i].size);
    if (!_Context->setOutputTensorAddress (
            tensorrt10_tensor_info.name, output[i].data)) {
      ml_loge ("Unable to set output tensor address");
      throw std::runtime_error ("Unable to set output tensor address");
    }
  }

  /* Execute the network */
  if (!_Context->enqueueV3 (_stream)) {
    ml_loge ("Failed to execute the network");
    throw std::runtime_error ("Failed to execute the network");
  }

  /* Wait for GPU to finish the inference */
  status = cudaStreamSynchronize (_stream);

  if (status != cudaSuccess) {
    ml_loge ("Failed to synchronize the cuda stream");
    throw std::runtime_error ("Failed to synchronize the cuda stream");
  }
}

/**
 * @brief Describe the subplugin's setting.
 */
void
tensorrt10_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = FALSE;
  info.allocate_in_invoke = TRUE;
  info.run_without_model = FALSE;
  info.verify_model_path = TRUE;
  info.hw_list = hw_list;
  info.num_hw = num_hw;
}

/**
 * @brief Get the in/output tensors info.
 */
int
tensorrt10_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops != GET_IN_OUT_INFO) {
    return -ENOENT;
  }

  gst_tensors_info_copy (std::addressof (in_info), std::addressof (_inputTensorMeta));
  gst_tensors_info_copy (std::addressof (out_info), std::addressof (_outputTensorMeta));

  return 0;
}

/**
 * @brief Override eventHandler to free Cuda data buffer.
 */
int
tensorrt10_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  if (ops == DESTROY_NOTIFY) {
    if (data.data != nullptr) {
      cudaFree (data.data);
    }
  }
  return 0;
}

/**
 * @brief Parses the model; changes the state of the provided parser.
 */
void
tensorrt10_subplugin::constructNetwork (NvInferUniquePtr<nvonnxparser::IParser> &parser) const
{
  auto parsed = parser->parseFromFile (
      _model_path, static_cast<int> (nvinfer1::ILogger::Severity::kWARNING));
  if (!parsed) {
    ml_loge ("Unable to parse onnx file");
    throw std::runtime_error ("Unable to parse onnx file");
  }
}

/**
 * Builds and saves the .engine file.
 */
void
tensorrt10_subplugin::buildSaveEngine () const
{
  auto builder = makeNvInferUniquePtr (nvinfer1::createInferBuilder (gLogger));
  if (!builder) {
    ml_loge ("Unable to create builder");
    throw std::runtime_error ("Unable to create builder");
  }

  auto network = makeNvInferUniquePtr (builder->createNetworkV2 (0));
  if (!network) {
    ml_loge ("Unable to create network");
    throw std::runtime_error ("Unable to create network");
  }

  auto config = makeNvInferUniquePtr (builder->createBuilderConfig ());
  if (!config) {
    ml_loge ("Unable to create builder config");
    throw std::runtime_error ("Unable to create builder config");
  }

  auto parser = makeNvInferUniquePtr (nvonnxparser::createParser (*network, gLogger));
  if (!parser) {
    ml_loge ("Unable to create onnx parser");
    throw std::runtime_error ("Unable to create onnx parser");
  }

  constructNetwork (parser);

  auto host_memory
      = makeNvInferUniquePtr (builder->buildSerializedNetwork (*network, *config));
  if (!host_memory) {
    ml_loge ("Unable to build serialized network");
    throw std::runtime_error ("Unable to build serialized network");
  }

  std::ofstream engineFile (_engine_fs_path, std::ios::binary);
  if (!engineFile) {
    ml_loge ("Unable to open engine file for saving");
    throw std::runtime_error ("Unable to open engine file for saving");
  }
  engineFile.write (static_cast<char *> (host_memory->data ()), host_memory->size ());
}

/**
 * @brief Loads the .engine model file and makes a member object to be used for inference.
 */
void
tensorrt10_subplugin::loadEngine ()
{
  // Create file
  std::ifstream file (_engine_fs_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg ();
  if (size < 0) {
    ml_loge ("Unable to open engine file %s", std::string (_engine_fs_path).data ());
    throw std::runtime_error ("Unable to open engine file");
  }

  file.seekg (0, std::ios::beg);
  ml_logi ("Loading tensorrt10 engine from file: %s with buffer size: %" G_GUINT64_FORMAT,
      std::string (_engine_fs_path).data (), size);

  // Read file
  std::vector<char> tensorrt10_engine_file_buffer (size);
  if (!file.read (tensorrt10_engine_file_buffer.data (), size)) {
    ml_loge ("Unable to read engine file %s", std::string (_engine_fs_path).data ());
    throw std::runtime_error ("Unable to read engine file");
  }

  // Create an engine, a representation of the optimized model.
  _Engine = NvInferUniquePtr<nvinfer1::ICudaEngine> (_Runtime->deserializeCudaEngine (
      tensorrt10_engine_file_buffer.data (), tensorrt10_engine_file_buffer.size ()));
  if (!_Engine) {
    ml_loge ("Unable to deserialize tensorrt10 engine");
    throw std::runtime_error ("Unable to deserialize tensorrt10 engine");
  }
}


/**
 * @brief Loads and interprets the model file.
 * @param[in] prop: property of tensor_filter instance
 */
void
tensorrt10_subplugin::loadModel (const GstTensorFilterProperties *prop)
{
  UNUSED (prop);

  // Set the device index
  ml_logd ("Loading model on device index: %d", _device_id);
  auto ret = cudaSetDevice (_device_id);
  if (ret != 0) {
    int num_gpus;
    cudaGetDeviceCount (&num_gpus);
    ml_loge ("Unable to set GPU device index to: %d. CUDA-capable GPU(s): %d.",
        _device_id, num_gpus);
    throw std::runtime_error ("Unable to set GPU device index");
  }

  checkUnifiedMemory ();

  // Parse model from .onnx and create .engine if necessary
  std::filesystem::path model_fs_path (_model_path);
  if (".onnx" == model_fs_path.extension ()) {
    _engine_fs_path = std::filesystem::path ("/tmp") / model_fs_path.stem ();
    _engine_fs_path += ".engine";
    if (!std::filesystem::exists (_engine_fs_path)) {
      buildSaveEngine ();
      g_assert (std::filesystem::exists (_engine_fs_path));
    }
  } else if (".engine" == model_fs_path.extension ()) {
    _engine_fs_path = model_fs_path;
  } else {
    ml_loge ("Unsupported model file extension %s",
        std::string (model_fs_path.extension ()).data ());
    throw std::runtime_error ("Unsupported model file extension");
  }

  // Create a runtime to deserialize the engine file.
  _Runtime = makeNvInferUniquePtr (nvinfer1::createInferRuntime (gLogger));
  if (!_Runtime) {
    ml_loge ("Failed to create TensorRT runtime");
    throw std::runtime_error ("Failed to create TensorRT runtime");
  }

  loadEngine ();

  /* Create ExecutionContext object */
  _Context = makeNvInferUniquePtr (_Engine->createExecutionContext ());
  if (!_Context) {
    ml_loge ("Failed to create the TensorRT ExecutionContext object");
    throw std::runtime_error ("Failed to create the TensorRT ExecutionContext object");
  }

  // Create the cuda stream
  cudaStreamCreate (&_stream);

  // Get number of IO buffers
  auto num_io_buffers = _Engine->getNbIOTensors ();
  if (num_io_buffers <= 0) {
    ml_loge ("Engine has no IO buffers");
    throw std::runtime_error ("Engine has no IO buffers");
  }

  // Iterate the model io buffers
  _tensorrt10_input_tensor_infos.clear ();
  _tensorrt10_output_tensor_infos.clear ();
  for (int buffer_index = 0; buffer_index < num_io_buffers; ++buffer_index) {
    NvInferTensorInfo tensorrt10_tensor_info{};

    // Get buffer name
    tensorrt10_tensor_info.name = _Engine->getIOTensorName (buffer_index);

    // Read and verify IO buffer shape
    tensorrt10_tensor_info.shape
        = _Engine->getTensorShape (tensorrt10_tensor_info.name);
    if (tensorrt10_tensor_info.shape.d[0] == -1) {
      ml_loge ("Dynamic batch size is not supported");
      throw std::runtime_error ("Dynamic batch size is not supported");
    }

    // Get data type and buffer size info
    tensorrt10_tensor_info.mode
        = _Engine->getTensorIOMode (tensorrt10_tensor_info.name);
    tensorrt10_tensor_info.dtype
        = _Engine->getTensorDataType (tensorrt10_tensor_info.name);
    tensorrt10_tensor_info.dtype_size
        = getTensorRTDataTypeSize (tensorrt10_tensor_info.dtype);
    tensorrt10_tensor_info.volume = getVolume (tensorrt10_tensor_info.shape);
    tensorrt10_tensor_info.buffer_size
        = tensorrt10_tensor_info.dtype_size * tensorrt10_tensor_info.volume;
    ml_logd ("BUFFER SIZE: %" G_GUINT64_FORMAT, tensorrt10_tensor_info.buffer_size);

    // Iterate the input and output buffers
    if (tensorrt10_tensor_info.mode == nvinfer1::TensorIOMode::kINPUT) {

      if (!_Context->setInputShape (
              tensorrt10_tensor_info.name, tensorrt10_tensor_info.shape)) {
        ml_loge ("Unable to set input shape");
        throw std::runtime_error ("Unable to set input shape");
      }

      // Allocate only for input, memory for output is allocated in the invoke method.
      allocBuffer (&tensorrt10_tensor_info.buffer, tensorrt10_tensor_info.buffer_size);
      if (!_Context->setInputTensorAddress (
              tensorrt10_tensor_info.name, tensorrt10_tensor_info.buffer)) {
        ml_loge ("Unable to set input tensor address");
        throw std::runtime_error ("Unable to set input tensor address");
      }

      _tensorrt10_input_tensor_infos.push_back (tensorrt10_tensor_info);

    } else if (tensorrt10_tensor_info.mode == nvinfer1::TensorIOMode::kOUTPUT) {

      _tensorrt10_output_tensor_infos.push_back (tensorrt10_tensor_info);

    } else {
      ml_loge ("TensorIOMode not supported");
      throw std::runtime_error ("TensorIOMode not supported");
    }
  }

  if (!_Context->allInputDimensionsSpecified ()) {
    ml_loge ("Not all required dimensions were specified");
    throw std::runtime_error ("Not all required dimensions were specified");
  }

  convertTensorsInfo (_tensorrt10_input_tensor_infos, _inputTensorMeta);
  convertTensorsInfo (_tensorrt10_output_tensor_infos, _outputTensorMeta);
}

/**
 * Converts the NvInferTensorInfo's to the nnstreamer GstTensorsInfo.
 */
void
tensorrt10_subplugin::convertTensorsInfo (
    const std::vector<NvInferTensorInfo> &tensorrt10_tensor_infos, GstTensorsInfo &info) const
{
  gst_tensors_info_init (std::addressof (info));
  info.num_tensors = tensorrt10_tensor_infos.size ();

  for (guint tensor_index = 0; tensor_index < info.num_tensors; ++tensor_index) {
    const auto &tensorrt10_tensor_info = tensorrt10_tensor_infos[tensor_index];

    // Set the nnstreamer GstTensorInfo properties
    GstTensorInfo *tensor_info
        = gst_tensors_info_get_nth_info (std::addressof (info), tensor_index);
    tensor_info->name = g_strdup (tensorrt10_tensor_info.name);
    tensor_info->type = getNnStreamerDataType (tensorrt10_tensor_info.dtype);

    // Set tensor dimensions in reverse order
    for (int dim_index = 0; dim_index < tensorrt10_tensor_info.shape.nbDims; ++dim_index) {
      std::size_t from_dim_index = tensorrt10_tensor_info.shape.nbDims - dim_index - 1;
      tensor_info->dimension[dim_index] = tensorrt10_tensor_info.shape.d[from_dim_index];
    }
  }
}

/**
 * @brief Return whether Unified Memory is supported or not.
 * @note After Cuda version 6, logical Unified Memory is supported in
 * programming language level. However, if the target device is not supported,
 * then cudaMemcpy() internally occurs and it makes performance degradation.
 */
void
tensorrt10_subplugin::checkUnifiedMemory () const
{
  int version;

  if (cudaRuntimeGetVersion (&version) != cudaSuccess) {
    ml_loge ("Unable to get cuda runtime version");
    throw std::runtime_error ("Unable to get cuda runtime version");
  }

  /* Unified memory requires at least CUDA-6 */
  if (version < 6000) {
    ml_loge ("Unified memory requires at least CUDA-6");
    throw std::runtime_error ("Unified memory requires at least CUDA-6");
  }

  // Get device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, _device_id);
  if (prop.managedMemory == 0) {
    ml_loge ("The current device does not support managedmemory");
    throw std::runtime_error ("The current device does not support managedmemory");
  }

  // The cuda programming guide specifies at least compute capability version 5
  //  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming
  if (prop.major < 5) {
    ml_loge ("The minimum required compute capability for unified memory is version 5");
    throw std::runtime_error (
        "The minimum required compute capability for unified memory is version 5");
  }
}

/**
 * @brief Allocates a GPU buffer memory
 * @param[out] buffer : pointer to allocated memory
 * @param[in] size : allocation size in bytes
 */
void
tensorrt10_subplugin::allocBuffer (void **buffer, gsize size)
{
  cudaError_t status = cudaMallocManaged (buffer, size);

  if (status != cudaSuccess) {
    ml_loge ("Failed to allocate Cuda memory");
    throw std::runtime_error ("Failed to allocate Cuda memory");
  }
}

/**
 * @brief Calculates the volume (in elements, not in bytes) of the provided shape.
 * @param[in] shape : The shape for which to calculate the volume.
 */
std::size_t
tensorrt10_subplugin::getVolume (const nvinfer1::Dims &shape) const
{
  auto volume = 1;
  for (auto i = 0; i < shape.nbDims; ++i) {
    volume *= shape.d[i];
  }
  return volume;
}

/**
 * @brief Get the size of the TensorRT data type.
 * @param[in] tensorrt10_data_type : The TensorRT data type.
 * @note: see also https://github.com/NVIDIA/TensorRT/blob/ccf119972b50299ba00d35d39f3938296e187f4e/samples/common/common.h#L539C1-L552C14
 */
std::size_t
tensorrt10_subplugin::getTensorRTDataTypeSize (nvinfer1::DataType tensorrt10_data_type) const
{
  switch (tensorrt10_data_type) {
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kINT4:
    default:
      ml_loge ("Element size is not implemented for data-type");
  }
  ml_loge ("Unable to determine tensorrt10 data type size");
  throw std::runtime_error ("Unable to determine tensorrt10 data type size");
}

/**
 * @brief Get the corresponding nnstreamer tensor_type based on the TensorRT data type.
 * @param[in] tensorrt10_data_type : The TensorRT data type.
 * @return The nnstreamer tensor_type.
 */
tensor_type
tensorrt10_subplugin::getNnStreamerDataType (nvinfer1::DataType tensorrt10_data_type) const
{
  switch (tensorrt10_data_type) {
    case nvinfer1::DataType::kINT64:
      return _NNS_INT64;
    case nvinfer1::DataType::kINT32:
      return _NNS_INT32;
    case nvinfer1::DataType::kFLOAT:
      return _NNS_FLOAT32;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF:
      return _NNS_FLOAT16;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
      return _NNS_UINT8;
    case nvinfer1::DataType::kINT8:
      return _NNS_INT8;
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kINT4:
    default:
      ml_loge ("Element size is not implemented for data type.");
  }
  ml_loge ("Unable to get the nnstreamer data type");
  throw std::runtime_error ("Unable to get the nnstreamer data type");
}

tensorrt10_subplugin *tensorrt10_subplugin::registeredRepresentation = nullptr;

/**
 * @brief Register the tensorrt10_subplugin object.
 */
void
tensorrt10_subplugin::init_filter_tensorrt10 (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<tensorrt10_subplugin> ();
  nnstreamer_filter_set_custom_property_desc (
      name, "DeviceId", "Set the GPU device ID to use (default: 0)", NULL);
}

/**
 * @brief Unregister the tensorrt10_subplugin object.
 */
void
tensorrt10_subplugin::fini_filter_tensorrt10 (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
_init_filter_tensorrt10 (void)
{
  tensorrt10_subplugin::init_filter_tensorrt10 ();
}

/** @brief Destruct the subplugin */
void
_fini_filter_tensorrt10 (void)
{
  tensorrt10_subplugin::fini_filter_tensorrt10 ();
}

} /* namespace tensorfilter_tensorrt10 */
} /* namespace nnstreamer */
