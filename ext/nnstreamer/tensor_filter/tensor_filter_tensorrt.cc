/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter, TensorRT Module
 * Copyright (C) 2020 Sangjung Woo <sangjung.woo@samsung.com>
 */
/**
 * @file   tensor_filter_tensorrt.cc
 * @date   23 Oct 2020
 * @brief  TensorRT module for tensor_filter gstreamer plugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Sangjung Woo <sangjung.woo@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (TensorRT) for tensor_filter.
 *
 * @note Only support engine file as inference model.
 */

#include <filesystem>
#include <fstream>
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
namespace tensorfilter_tensorrt
{
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void _init_filter_tensorrt (void) __attribute__ ((constructor));
void _fini_filter_tensorrt (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

struct NvInferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using NvInferUniquePtr = std::unique_ptr<T, NvInferDeleter>;

template <typename T> NvInferUniquePtr<T> makeNvInferUniquePtr (T *t)
{
  return NvInferUniquePtr<T>{ t };
}

struct NvInferTensorInfo {
  const char* name;
  nvinfer1::TensorIOMode mode;
  nvinfer1::Dims shape;
  nvinfer1::DataType dtype;
  std::size_t dtype_size;
  std::size_t volume;
  std::size_t buffer_size;
  void *buffer; /**< Cuda buffer */
};

/** @brief tensorrt subplugin class */
class tensorrt_subplugin final : public tensor_filter_subplugin
{
  template <typename T> using UniquePtr = std::unique_ptr<T>;

  public:
  static void init_filter_tensorrt ();
  static void fini_filter_tensorrt ();

  tensorrt_subplugin ();
  ~tensorrt_subplugin ();

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
  static tensorrt_subplugin *registeredRepresentation;

  bool _configured{}; /**< Flag to keep track of whether this instance has been configured or not. */
  gchar *_model_path{}; /**< engine file path to infer */
  std::filesystem::path _engine_fs_path{};
  cudaStream_t _stream{}; /**< The cuda inference stream */

  GstTensorsInfo _inputTensorMeta;
  GstTensorsInfo _outputTensorMeta;

  NvInferUniquePtr<nvinfer1::IRuntime> _Runtime{};
  NvInferUniquePtr<nvinfer1::ICudaEngine> _Engine{};
  NvInferUniquePtr<nvinfer1::IExecutionContext> _Context{};

  std::vector<NvInferTensorInfo> _tensorrt_input_tensor_infos{};
  std::vector<NvInferTensorInfo> _tensorrt_output_tensor_infos{};

  void cleanup();
  int allocBuffer (void **buffer, gsize size);
  int loadModel (const GstTensorFilterProperties *prop);
  int checkUnifiedMemory () const;
  void convertTensorsInfo (const std::vector<NvInferTensorInfo>& tensorrt_tensor_infos, GstTensorsInfo &info, bool reverseDims = true) const;
  std::uint32_t getVolume (const nvinfer1::Dims& shape) const;
  std::uint32_t getTensorRTDataTypeSize (nvinfer1::DataType tensorrt_data_type) const;
  tensor_type getNnStreamerDataType(nvinfer1::DataType tensorrt_data_type) const;

  int constructNetwork(
  /*  NvInferUniquePtr<nvinfer1::IBuilder>& builder,
    NvInferUniquePtr<nvinfer1::INetworkDefinition>& network,
    NvInferUniquePtr<nvinfer1::IBuilderConfig>& config, */
    NvInferUniquePtr<nvonnxparser::IParser>& parser) const;
  int buildSaveEngine () const;
  int loadEngine ();
};

const char *tensorrt_subplugin::name = "tensorrt";
const accl_hw tensorrt_subplugin::hw_list[] = {};

/**
 * @brief constructor of tensorrt_subplugin
 */
tensorrt_subplugin::tensorrt_subplugin ()
    : tensor_filter_subplugin (), _model_path (nullptr)
{
}

/**
 * @brief destructor of tensorrt_subplugin
 */
tensorrt_subplugin::~tensorrt_subplugin ()
{
  cleanup ();
}

void
tensorrt_subplugin::cleanup ()
{
  if (!_configured)
    return;

  gst_tensors_info_free (&_inputTensorMeta);
  gst_tensors_info_free (&_outputTensorMeta);

  for (const auto& tensorrt_tensor_info : _tensorrt_input_tensor_infos) {
    cudaFree (tensorrt_tensor_info.buffer);
  }

  if (_model_path != nullptr)
    g_free (_model_path);

  if (_stream) {
    cudaStreamDestroy(_stream);
  }
}

/**
 * @brief Returns an empty instance.
 * @return an empty instance
 */
tensor_filter_subplugin &
tensorrt_subplugin::getEmptyInstance ()
{
  return *(new tensorrt_subplugin ());
}

/**
 * @brief Configure the instance of the tensorrt_subplugin.
 * @param[in] prop property of tensor_filter instance
 */
void
tensorrt_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  /* Set model path */
  if (prop->num_models != 1 || !prop->model_files[0]) {
    ml_loge ("TensorRT filter requires one engine model file.");
    throw std::invalid_argument ("The .engine model file is not given.");
  }
  assert (_model_path == nullptr);
  _model_path = g_strdup (prop->model_files[0]);

  /* Make a TensorRT engine */
  if (loadModel (prop)) {
    ml_loge ("Failed to build a TensorRT engine");
    throw std::runtime_error ("Failed to build a TensorRT engine");
  }

  _configured = true;
}

/**
 * @brief Invoke the TensorRT model and get the inference result.
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 */
void
tensorrt_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  g_assert (_configured);

  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  cudaError_t status;

  /* Copy input data to Cuda memory space */
  for (std::size_t i = 0; i < _tensorrt_input_tensor_infos.size(); ++i) {
    const auto& tensorrt_tensor_info = _tensorrt_input_tensor_infos[i];
    g_assert (tensorrt_tensor_info.buffer_size == input[i].size);

    status = cudaMemcpyAsync(
      tensorrt_tensor_info.buffer,
      input[i].data,
      input[i].size,
      cudaMemcpyHostToDevice,
      _stream
    );

    if (status != cudaSuccess) {
      ml_loge ("Failed to copy to cuda input buffer");
      throw std::runtime_error ("Failed to copy to cuda input buffer");
    }
    // memcpy (tensorrt_tensor_info.buffer, input[i].data, input[i].size);
  }

  /* Allocate output buffer */
  // for (std::size_t i = 0; i < _tensorrt_output_tensor_infos.size(); ++i) {
  //   const auto& tensorrt_tensor_info = _tensorrt_output_tensor_infos[i];
  //   g_assert (tensorrt_tensor_info.buffer_size == output[i].size);
  //   if (allocBuffer (&output[i].data, output[i].size) != 0) {
  //     ml_loge ("Failed to allocate GPU memory for output");
  //     throw std::runtime_error ("Failed to allocate GPU memory for output");
  //   }

  //   if (!_Context->setOutputTensorAddress(tensorrt_tensor_info.name, output[i].data)) {
  //     ml_loge ("Unable to set output tensor address");
  //     throw std::runtime_error ("Unable to set output tensor address");
  //   }
  // }

  /* Execute the network */
  if (!_Context->enqueueV3 (_stream)) {
    ml_loge ("Failed to execute the network");
    throw std::runtime_error ("Failed to execute the network");
  }

  for (std::size_t i = 0; i < _tensorrt_output_tensor_infos.size(); ++i) {
    const auto& tensorrt_tensor_info = _tensorrt_output_tensor_infos[i];
    g_assert (tensorrt_tensor_info.buffer_size == output[i].size);

    status = cudaMemcpyAsync(
      output[i].data,
      tensorrt_tensor_info.buffer,
      output[i].size,
      cudaMemcpyDeviceToHost,
      _stream
    );

    if (status != cudaSuccess) {
      ml_loge ("Failed to copy from cuda output buffer");
      throw std::runtime_error ("Failed to copy from cuda output buffer");
    }
  }

  /* Wait for GPU to finish the inference */
  status = cudaStreamSynchronize(_stream);

  if (status != cudaSuccess) {
    ml_loge ("Failed to synchronize the cuda stream");
    throw std::runtime_error ("Failed to synchronize the cuda stream");
  }
}

/**
 * @brief Describe the subplugin's setting.
 */
void
tensorrt_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
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
 * @brief Get the in/output tensors info.
 */
int
tensorrt_subplugin::getModelInfo (
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
tensorrt_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  if (ops == DESTROY_NOTIFY) {
    if (data.data != nullptr) {
      cudaFree (data.data);
    }
  }
  return 0;
}

int
tensorrt_subplugin::constructNetwork(
/*  NvInferUniquePtr<nvinfer1::IBuilder>& builder,
  NvInferUniquePtr<nvinfer1::INetworkDefinition>& network,
  NvInferUniquePtr<nvinfer1::IBuilderConfig>& config, */
  NvInferUniquePtr<nvonnxparser::IParser>& parser) const
{
  auto parsed = parser->parseFromFile (
    _model_path,
    static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
  if (!parsed) {
    ml_loge ("Unable to parse onnx file");
    return -1;
  }

  // todo: add support for builder parameters
  // if (mParams.fp16)
  // {
  //     config->setFlag(BuilderFlag::kFP16);
  // }
  // if (mParams.bf16)
  // {
  //     config->setFlag(BuilderFlag::kBF16);
  // }
  // if (mParams.int8)
  // {
  //     config->setFlag(BuilderFlag::kINT8);
  //     samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
  // }

  // samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

  return 0;
}

/***/
int
tensorrt_subplugin::buildSaveEngine () const
{
  auto builder = makeNvInferUniquePtr (nvinfer1::createInferBuilder (gLogger));
  if (!builder) {
    ml_loge ("Unable to create builder");
    return -1;
  }

  auto network = makeNvInferUniquePtr (builder->createNetworkV2 (0));
  if (!network) {
    ml_loge ("Unable to create network");
    return -1;
  }

  auto config = makeNvInferUniquePtr (builder->createBuilderConfig ());
  if (!config) {
    ml_loge ("Unable to create builder config");
    return -1;
  }

  auto parser = makeNvInferUniquePtr (nvonnxparser::createParser (*network, gLogger));
  if (!parser) {
    ml_loge ("Unable to create onnx parser");
    return -1;
  }

  if (constructNetwork(
    /* builder,
    network,
    config, */
    parser) != 0) {
    return -1;
  }

  // // CUDA stream used for profiling by the builder.
  // auto profileStream = samplesCommon::makeCudaStream();
  // if (!profileStream)
  // {
  //     return -1;
  // }
  // config->setProfileStream(*profileStream);

  auto host_memory = makeNvInferUniquePtr (builder->buildSerializedNetwork (*network, *config));
  if (!host_memory) {
    ml_loge ("Unable to build serialized network");
    return -1;
  }

  std::ofstream engineFile(_engine_fs_path, std::ios::binary);
  if (!engineFile) {
    ml_loge ("Unable to open engine file for saving");
    return -1;
  }
  engineFile.write(static_cast<char*>(host_memory->data()), host_memory->size());

  return 0;
}

/**
 * @brief Load .engine model file and make internal object for TensorRT inference
 * @return 0 if successfully loaded, or -1.
 */
int
tensorrt_subplugin::loadEngine ()
{
  // Create file
  std::ifstream file(_engine_fs_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  if (size < 0) {
    ml_loge ("Unable to open engine file %s", std::string(_engine_fs_path).data());
    return -1;
  }

  file.seekg(0, std::ios::beg);
  ml_logi(
    "Loading tensorrt engine from file: %s with buffer size: %" G_GUINT64_FORMAT,
    std::string(_engine_fs_path).data(),
    size
  );

  // Read file
  std::vector<char> tensorrt_engine_file_buffer(size);
  if (!file.read(tensorrt_engine_file_buffer.data(), size)) {
    ml_loge ("Unable to read engine file %s", std::string(_engine_fs_path).data());
    return -1;
  }

  // Create an engine, a representation of the optimized model.
  _Engine = NvInferUniquePtr<nvinfer1::ICudaEngine>(
    _Runtime->deserializeCudaEngine(
      tensorrt_engine_file_buffer.data(),
      tensorrt_engine_file_buffer.size()
    )
  );
  if (!_Engine) {
    ml_loge ("Unable to deserialize tensorrt engine");
    return -1;
  }

  return 0;
}


/**
 * @brief Load and interpret model file.
 * @param[in] prop : property of tensor_filter instance
 * @return 0 if successfully loaded, or -1.
 */
int
tensorrt_subplugin::loadModel (const GstTensorFilterProperties *prop)
{
  // GstTensorInfo *_info;

  UNUSED (prop);

  if (checkUnifiedMemory () != 0) {
    ml_loge ("Failed to enable unified memory");
    return -1;
  }

  // Set the device index
  // todo: read device_id from props?
  auto device_id = 0;
  auto ret = cudaSetDevice (device_id);
  if (ret != 0) {
      int num_gpus;
      cudaGetDeviceCount (&num_gpus);
      ml_loge ("Unable to set GPU device index to: %d. CUDA-capable GPU(s): %d.",
        device_id,
        num_gpus
      );
      return -1;
  }

  // Parse model from .onnx and create .engine if necessary
  std::filesystem::path model_fs_path(_model_path);
  if (".onnx" == model_fs_path.extension ()) {
    _engine_fs_path = std::filesystem::path("/tmp") / model_fs_path.stem();
    _engine_fs_path += ".engine";
    if (!std::filesystem::exists(_engine_fs_path)) {
      buildSaveEngine ();
    }
    if (!std::filesystem::exists(_engine_fs_path)) {
      ml_loge ("Unable to build and/or save engine file");
      return -1;
    }
  } else if (".engine" == model_fs_path.extension ()) {
    _engine_fs_path = model_fs_path;
  } else {
    ml_loge ("Unsupported model file extension %s", std::string (model_fs_path.extension ()).data());
    return -1;
  }

  // Create a runtime to deserialize the engine file.
  _Runtime = makeNvInferUniquePtr (nvinfer1::createInferRuntime (gLogger));
  if (!_Runtime) {
    ml_loge ("Failed to create TensorRT runtime");
    return -1;
  }

  if (loadEngine () != 0) {
    return -1;
  }

  /* Create ExecutionContext object */
  _Context = makeNvInferUniquePtr (_Engine->createExecutionContext ());
  if (!_Context) {
    ml_loge ("Failed to create the TensorRT ExecutionContext object");
    return -1;
  }

  // Create the cuda stream
  cudaStreamCreate(&_stream);

  // Set optimization profile to the default
  // todo: support for optimization profiles
  // if (!_Context->setOptimizationProfileAsync(0, stream)) {
  //   ml_loge ("Unable to set optimization profile");
  //   return -1;
  // }

  // Get number of IO buffers
  auto num_io_buffers = _Engine->getNbIOTensors();
  if (num_io_buffers <= 0) {
    ml_loge ("Engine has no IO buffers");
    return -1;
  }

  // Iterate the model io buffers
  _tensorrt_input_tensor_infos.clear();
  _tensorrt_output_tensor_infos.clear();
  for (int buffer_index = 0; buffer_index < num_io_buffers; ++buffer_index) {
    NvInferTensorInfo tensorrt_tensor_info{};

    // Get buffer name
    tensorrt_tensor_info.name = _Engine->getIOTensorName(buffer_index);

    // Read and verify IO buffer shape
    tensorrt_tensor_info.shape = _Engine->getTensorShape(tensorrt_tensor_info.name);
    if (tensorrt_tensor_info.shape.d[0] == -1) {
      ml_loge ("Dynamic batch size is not supported");
      return -1;
    }

    // Get data type and buffer size info
    tensorrt_tensor_info.mode = _Engine->getTensorIOMode(tensorrt_tensor_info.name);
    tensorrt_tensor_info.dtype = _Engine->getTensorDataType(tensorrt_tensor_info.name);
    tensorrt_tensor_info.dtype_size = getTensorRTDataTypeSize(tensorrt_tensor_info.dtype);
    tensorrt_tensor_info.volume = getVolume(tensorrt_tensor_info.shape);
    tensorrt_tensor_info.buffer_size = tensorrt_tensor_info.dtype_size * tensorrt_tensor_info.volume;
    ml_logd ("BUFFER SIZE: %" G_GUINT64_FORMAT, tensorrt_tensor_info.buffer_size);

    if (allocBuffer (&tensorrt_tensor_info.buffer, tensorrt_tensor_info.buffer_size) != 0) {
      ml_loge ("Failed to allocate GPU memory for tensorrt tensor");
      return -1;
    }

    // Allocate GPU memory for input and output buffers
    if (tensorrt_tensor_info.mode == nvinfer1::TensorIOMode::kINPUT) {
      if (!_Context->setInputShape(tensorrt_tensor_info.name, tensorrt_tensor_info.shape)) {
        ml_loge ("Unable to set input shape");
        return -1;
      }

      if (!_Context->setInputTensorAddress(tensorrt_tensor_info.name, tensorrt_tensor_info.buffer)) {
        ml_loge ("Unable to set input tensor address");
        return -1;
      }

      _tensorrt_input_tensor_infos.push_back(tensorrt_tensor_info);

    } else if (tensorrt_tensor_info.mode == nvinfer1::TensorIOMode::kOUTPUT) {

      if (!_Context->setOutputTensorAddress(tensorrt_tensor_info.name, tensorrt_tensor_info.buffer)) {
        ml_loge ("Unable to set output tensor address");
        return -1;
      }

      _tensorrt_output_tensor_infos.push_back(tensorrt_tensor_info);

    } else {
      ml_loge ("TensorIOMode not supported");
      return -1;
    }
  }

  if (!_Context->allInputDimensionsSpecified()) {
    ml_loge ("Not all required dimensions were specified");
    return -1;
  }

  convertTensorsInfo(_tensorrt_input_tensor_infos, _inputTensorMeta);
  convertTensorsInfo(_tensorrt_output_tensor_infos, _outputTensorMeta);

  return 0;
}

void
tensorrt_subplugin::convertTensorsInfo (
  const std::vector<NvInferTensorInfo>& tensorrt_tensor_infos,
  GstTensorsInfo &info,
  bool reverseDims /* = true */) const
{
  gst_tensors_info_init (std::addressof (info));
  info.num_tensors = tensorrt_tensor_infos.size();

  for (guint tensor_index = 0; tensor_index < info.num_tensors; ++tensor_index) {
    const auto& tensorrt_tensor_info = tensorrt_tensor_infos[tensor_index];

    // Set the nnstreamer GstTensorInfo properties
    GstTensorInfo *tensor_info = gst_tensors_info_get_nth_info (std::addressof (info), tensor_index);
    tensor_info->name = g_strdup(tensorrt_tensor_info.name);
    tensor_info->type = getNnStreamerDataType(tensorrt_tensor_info.dtype);
    // Set tensor dimensions in reverse order
    for (int dim_index = 0; dim_index < tensorrt_tensor_info.shape.nbDims; ++dim_index) {
      std::size_t from_dim_index = dim_index;
      if (reverseDims) {
        from_dim_index = tensorrt_tensor_info.shape.nbDims - dim_index - 1;
      }
      tensor_info->dimension[dim_index] = tensorrt_tensor_info.shape.d[from_dim_index];
    }
  }
}

/**
 * @brief Return whether Unified Memory is supported or not.
 * @return 0 if Unified Memory is supported. non-zero if error.
 * @note After Cuda version 6, logical Unified Memory is supported in
 * programming language level. However, if the target device is not supported,
 * then cudaMemcpy() internally occurs and it makes performance degradation.
 */
int
tensorrt_subplugin::checkUnifiedMemory () const
{
  int version;

  if (cudaRuntimeGetVersion (&version) != cudaSuccess)
    return -1;

  /* Unified memory requires at least CUDA-6 */
  if (version < 6000)
    return -1;

  return 0;
}

/**
 * @brief Allocate a GPU buffer memory
 * @param[out] buffer : pointer to allocated memory
 * @param[in] size : allocation size in bytes
 * @return 0 if OK. non-zero if error.
 */
int
tensorrt_subplugin::allocBuffer (void **buffer, gsize size)
{
  cudaError_t status = cudaMallocManaged (buffer, size);

  if (status != cudaSuccess) {
    ml_loge ("Failed to allocate Cuda memory");
    return -1;
  }

  return 0;
}

std::uint32_t
tensorrt_subplugin::getVolume (const nvinfer1::Dims& shape) const
{
    auto volume = 1;
    for (auto i = 0; i < shape.nbDims; ++i) {
        volume *= shape.d[i];
    }
    return volume;
}

/**
 * @brief Get the size of the TensorRT data type.
 * @param[in] tensorrt_data_type : The TensorRT data type.
 * @return The size if OK. -1 if the TensorRT data type is not supported.
 */
std::uint32_t
tensorrt_subplugin::getTensorRTDataTypeSize (nvinfer1::DataType tensorrt_data_type) const
{
  // Unfortunately, no implementation seems to be provided in the C++ api.
  // https://github.com/NVIDIA/TensorRT/blob/a1820ecdf6c14b7e50d2feb4fff4cc149dd6846b/python/packaging/bindings_wheel/tensorrt/__init__.py
  switch (tensorrt_data_type) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kHALF:
      return sizeof(float) / 2;
    case nvinfer1::DataType::kINT8:
      return sizeof(std::int8_t);
    case nvinfer1::DataType::kINT32:
      return sizeof(std::int32_t);
    case nvinfer1::DataType::kBOOL:
      return sizeof(bool);
    case nvinfer1::DataType::kUINT8:
      return sizeof(std::uint8_t);
    // kFP8 not supported yet by tensorrt
    case nvinfer1::DataType::kFP8:
      return 1;
    default:
      ml_loge ("TensorRT data type not supported.");
      return -1;
  }
  // mapping = {
  //     float32: 4,
  //     float16: 2,
  //     int8: 1,
  //     int32: 4,
  //     bool: 1,
  //     uint8: 1,
  //     fp8: 1,
  // }
}

tensor_type tensorrt_subplugin::getNnStreamerDataType(nvinfer1::DataType tensorrt_data_type) const
{
  switch (tensorrt_data_type) {
    case nvinfer1::DataType::kFLOAT:
      return _NNS_FLOAT32;
    case nvinfer1::DataType::kHALF:
      return _NNS_FLOAT16;
    case nvinfer1::DataType::kINT8:
      return _NNS_INT8;
    case nvinfer1::DataType::kINT32:
      return _NNS_INT32;
    case nvinfer1::DataType::kBOOL:
      return _NNS_INT8;
    case nvinfer1::DataType::kUINT8:
      return _NNS_UINT8;
    // kFP8 not supported yet by tensorrt
    case nvinfer1::DataType::kFP8:
    default:
      throw std::runtime_error("TensorRT data type not supported.");
  }

  // typedef enum _nns_tensor_type
  // {
  //   _NNS_INT32 = 0,
  //   _NNS_UINT32,
  //   _NNS_INT16,
  //   _NNS_UINT16,
  //   _NNS_INT8,
  //   _NNS_UINT8,
  //   _NNS_FLOAT64,
  //   _NNS_FLOAT32,
  //   _NNS_INT64,
  //   _NNS_UINT64,
  //   _NNS_FLOAT16, /**< added with nnstreamer 2.1.1-devel. If you add any operators (e.g., tensor_transform) to float16, it will either be not supported or be too inefficient. */
  //   _NNS_END,
  // } tensor_type;
}


tensorrt_subplugin *tensorrt_subplugin::registeredRepresentation = nullptr;

/**
 * @brief Register the tensorrt_subplugin object.
 */
void
tensorrt_subplugin::init_filter_tensorrt (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<tensorrt_subplugin> ();
}

/**
 * @brief Unregister the tensorrt_subplugin object.
 */
void
tensorrt_subplugin::fini_filter_tensorrt (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
_init_filter_tensorrt (void)
{
  tensorrt_subplugin::init_filter_tensorrt ();
}

/** @brief Destruct the subplugin */
void
_fini_filter_tensorrt (void)
{
  tensorrt_subplugin::fini_filter_tensorrt ();
}

} /* namespace tensorfilter_tensorrt */
} /* namespace nnstreamer */
