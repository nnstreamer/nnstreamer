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
        ml_logw ("%s", msg);
        break;
      case Severity::kINFO:
        ml_logi ("%s", msg);
        break;
      case Severity::kVERBOSE:
        ml_logd ("%s", msg);
        break;
      default:
        ml_loge ("%s", msg);
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

  gchar *_engine_path; /**< engine file path to infer */
  void *_inputBuffer; /**< Input Cuda buffer */
  void *_outputBuffer; /**< Output Cuda buffer */
  cudaStream_t _stream; /**< The cuda inference stream */

  GstTensorsInfo _inputTensorMeta;
  GstTensorsInfo _outputTensorMeta;

  nvinfer1::Dims _InputDims;
  nvinfer1::DataType _DataType;
  UniquePtr<nvinfer1::IRuntime> _Runtime{ nullptr };
  UniquePtr<nvinfer1::ICudaEngine> _Engine{ nullptr };
  UniquePtr<nvinfer1::IExecutionContext> _Context{ nullptr };

  int allocBuffer (void **buffer, gsize size);
  int setInputDims (guint input_rank);
  int setTensorType (tensor_type t);
  int loadModel (const GstTensorFilterProperties *prop);
  int checkUnifiedMemory ();
  std::uint32_t getVolume (const nvinfer1::Dims& shape) const;
  std::uint32_t getTensorRTDataTypeSize (nvinfer1::DataType tensorrt_data_type) const;

  /** @brief Make unique pointer */
  template <typename T> UniquePtr<T> makeUnique (T *t)
  {
    return UniquePtr<T>{ t };
  }
};

const char *tensorrt_subplugin::name = "tensorrt";
const accl_hw tensorrt_subplugin::hw_list[] = {};

/**
 * @brief constructor of tensorrt_subplugin
 */
tensorrt_subplugin::tensorrt_subplugin ()
    : tensor_filter_subplugin (), _engine_path (nullptr), _inputBuffer (nullptr)
{
  gst_tensors_info_init (&_inputTensorMeta);
  gst_tensors_info_init (&_outputTensorMeta);
}

/**
 * @brief destructor of tensorrt_subplugin
 */
tensorrt_subplugin::~tensorrt_subplugin ()
{
  gst_tensors_info_free (&_inputTensorMeta);
  gst_tensors_info_free (&_outputTensorMeta);

  if (_inputBuffer != nullptr)
    cudaFree (_inputBuffer);

  if (_engine_path != nullptr)
    g_free (_engine_path);

  cudaStreamDestroy(_stream);
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
  GstTensorInfo *_info;

  /* Set model path */
  if (prop->num_models != 1 || !prop->model_files[0]) {
    ml_loge ("TensorRT filter requires one engine model file.");
    throw std::invalid_argument ("The .engine model file is not given.");
  }
  assert (_engine_path == nullptr);
  _engine_path = g_strdup (prop->model_files[0]);

  /* Set input/output TensorsInfo */
  gst_tensors_info_copy (&_inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&_outputTensorMeta, &prop->output_meta);

  /* Set nvinfer1::Dims object */
  if (setInputDims (prop->input_ranks[0]) != 0) {
    throw std::invalid_argument ("TensorRT filter only supports 2, 3, 4 dimensions");
  }

  /* Get the datatype of input tensor */
  _info = gst_tensors_info_get_nth_info (&_inputTensorMeta, 0U);
  if (setTensorType (_info->type) != 0) {
    ml_loge ("TensorRT filter does not support the input data type.");
    throw std::invalid_argument ("TensorRT filter does not support the input data type.");
  }

  /* Make a TensorRT engine */
  if (loadModel (prop)) {
    ml_loge ("Failed to build a TensorRT engine");
    throw std::runtime_error ("Failed to build a TensorRT engine");
  }
}

/**
 * @brief Invoke the TensorRT model and get the inference result.
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 */
void
tensorrt_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  /* If internal _inputBuffer is nullptr, tne allocate GPU memory */
  // if (!_inputBuffer) {
  //   if (allocBuffer (&_inputBuffer, input->size) != 0) {
  //     ml_loge ("Failed to allocate GPU memory for input");
  //     throw std::runtime_error ("Failed to allocate GPU memory for input");
  //   }
  // }

  /* Copy input data to Cuda memory space */
  memcpy (_inputBuffer, input->data, input->size);

  /* Allocate output buffer */
  // if (allocBuffer (&output->data, output->size) != 0) {
  //   ml_loge ("Failed to allocate GPU memory for output");
  //   throw std::runtime_error ("Failed to allocate GPU memory for output");
  // }

  /* Bind the input and execute the network */
  std::vector<void *> bindings = { _inputBuffer, output->data };
  if (!_Context->enqueueV3 (_stream)) {
    ml_loge ("Failed to execute the network");
    throw std::runtime_error ("Failed to execute the network");
  }

  /* wait for GPU to finish the inference */
  cudaStreamSynchronize(_stream);
}

/**
 * @brief Describe the subplugin's setting.
 */
void
tensorrt_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = FALSE;
  info.allocate_in_invoke = TRUE;
  info.run_without_model = FALSE;
  info.verify_model_path = FALSE;
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

/**
 * @brief Load .engine model file and make internal object for TensorRT inference
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

  // Create a runtime to deserialize the engine file.
  _Runtime = makeUnique (nvinfer1::createInferRuntime (gLogger));
  if (!_Runtime) {
    ml_loge ("Failed to create TensorRT runtime");
    return -1;
  }

  // Create file
  std::ifstream file(_engine_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  ml_logw(
    "Loading tensorrt engine from file: %s with buffer size: %" G_GUINT64_FORMAT,
    _engine_path,
    size
  );

  // Read file
  std::vector<char> tensorrt_engine_file_buffer(size);
  if (!file.read(tensorrt_engine_file_buffer.data(), size)) {
    ml_loge ("Unable to read engine file %s", _engine_path);
    return -1;
  }

  // Create an engine, a representation of the optimized model.
  _Engine = makeUnique(
    _Runtime->deserializeCudaEngine(
      tensorrt_engine_file_buffer.data()
      , tensorrt_engine_file_buffer.size()
    )
  );
  if (!_Engine) {
    ml_loge ("Unable to deserialize tensorrt engine");
    return -1;
  }

  // Get number of IO buffers
  auto num_io_buffers = _Engine->getNbIOTensors();
  if (num_io_buffers <= 0) {
    ml_loge ("Engine has no IO buffers");
    return -1;
  }
  if (num_io_buffers != 2) {
    ml_loge ("Number of IO buffers is not supported (only 2 IO buffers are supported, 1 for input, 1 for output)");
    return -1;
  }

  // Iterate the model io buffers
  for (int buffer_index = 0; buffer_index < num_io_buffers; ++buffer_index) {
    // Get buffer name
    const auto tensor_name = _Engine->getIOTensorName(buffer_index);

    // Read and verify IO buffer shape
    const auto tensor_shape = _Engine->getTensorShape(tensor_name);
    if (tensor_shape.d[0] == -1) {
      ml_loge ("Dynamic batch size is not supported");
      return -1;
    }
    // if (tensor_shape.d[0] != 1) {
    //   ml_loge ("Only an explicit batch size of 1 is supported");
    //   return -1;
    // }

    // Does not work: "Get bytes per component failed, internalGetBindingVectorizedDim() result shall not equal -1."
    // const auto tensor_dtype_size = _Engine->getTensorBytesPerComponent(tensor_name);

    // Get data type and buffer size info
    const auto tensor_dtype = _Engine->getTensorDataType(tensor_name);
    const auto tensor_dtype_size = getTensorRTDataTypeSize(tensor_dtype);
    const auto tensor_volume = getVolume(tensor_shape);
    const auto tensor_buffer_size = tensor_dtype_size * tensor_volume;
    ml_logw ("BUFFER SIZE: %d", tensor_buffer_size);

    // Allocate GPU memory for input and output buffers
    const auto tensor_mode = _Engine->getTensorIOMode(tensor_name);
    if (tensor_mode == nvinfer1::TensorIOMode::kINPUT) {
      if (allocBuffer (&_inputBuffer, tensor_buffer_size) != 0) {
        ml_loge ("Failed to allocate GPU memory for input");
        return -1;
      }
    } else if (tensor_mode == nvinfer1::TensorIOMode::kOUTPUT) {
      if (allocBuffer (&_outputBuffer, tensor_buffer_size) != 0) {
        ml_loge ("Failed to allocate GPU memory for output");
        return -1;
      }
    } else {
      ml_loge ("TensorIOMode not supported");
      return -1;
    }
  }

  /* Create ExecutionContext obejct */
  _Context = makeUnique (_Engine->createExecutionContext ());
  if (!_Context) {
    ml_loge ("Failed to create the TensorRT ExecutionContext object");
    return -1;
  }

  cudaStreamCreate(&_stream);

  // const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  // auto network = makeUnique (builder->createNetworkV2 (explicitBatch));
  // if (!network) {
  //   ml_loge ("Failed to create network");
  //   return -1;
  // }

  // auto config = makeUnique (builder->createBuilderConfig ());
  // if (!config) {
  //   ml_loge ("Failed to create config");
  //   return -1;
  // }

  // auto parser = makeUnique (nvuffparser::createUffParser ());
  // if (!parser) {
  //   ml_loge ("Failed to create parser");
  //   return -1;
  // }

  // /* Register tensor input & output */
  // _info = gst_tensors_info_get_nth_info (&_inputTensorMeta, 0U);
  // parser->registerInput (_info->name, _InputDims, nvuffparser::UffInputOrder::kNCHW);

  // _info = gst_tensors_info_get_nth_info (&_outputTensorMeta, 0U);
  // parser->registerOutput (_info->name);

  // /* Parse the imported model */
  // parser->parse (_engine_path, *network, _DataType);

  // /* Set config */
  // builder->setMaxBatchSize (1);
  // config->setMaxWorkspaceSize (1 << 20);
  // config->setFlag (nvinfer1::BuilderFlag::kGPU_FALLBACK);

  // /* Create Engine object */
  // _Engine = makeUnique (builder->buildEngineWithConfig (*network, *config));
  // if (!_Engine) {
  //   ml_loge ("Failed to create the TensorRT Engine object");
  //   return -1;
  // }

  return 0;
}

/**
 * @brief Return whether Unified Memory is supported or not.
 * @return 0 if Unified Memory is supported. non-zero if error.
 * @note After Cuda version 6, logical Unified Memory is supported in
 * programming language level. However, if the target device is not supported,
 * then cudaMemcpy() internally occurs and it makes performance degradation.
 */
int
tensorrt_subplugin::checkUnifiedMemory (void)
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

/**
 * @brief Set the data type of the the TensorRT.
 * @param[in] t : data type of NNStreamer
 * @return 0 if OK. -1 if it is not supported.
 * @note TensorRT supports kFLOAT(32bit), kHALF(16bit), kINT8, kINT32 and kBOOL.
 *       However, NNStreamer does not support kHALF and kBOOL so error(i.e. -1) returns in this case.
 */
int
tensorrt_subplugin::setTensorType (tensor_type t)
{
  switch (t) {
    case _NNS_INT32:
      _DataType = nvinfer1::DataType::kINT32;
      break;
    case _NNS_FLOAT32:
      _DataType = nvinfer1::DataType::kFLOAT;
      break;
    case _NNS_INT8:
      _DataType = nvinfer1::DataType::kINT8;
      break;

    default:
      /**
       * TensorRT supports kFLOAT(32bit), kHALF(16bit), kINT8, kINT32 and kBOOL.
       * However, NNStreamer does not support kHALF and kBOOL.
       */
      return -1;
  }

  return 0;
}

/**
 * @brief Set the input dimension of the TensorRT.
 * @param[in] input_rank : actual input rank of input tensors.
 * @return 0 if OK. -1 if it is not supported.
 * @note The actual rank count is dependent on the dimension string of the tensor filter parameter.
 */
int
tensorrt_subplugin::setInputDims (guint input_rank)
{
  GstTensorInfo *_info;

  _info = gst_tensors_info_get_nth_info (&_inputTensorMeta, 0U);

  switch (input_rank) {
    case 2:
      _InputDims
          = nvinfer1::Dims2 ((int) _info->dimension[1], (int) _info->dimension[0]);
      break;

    case 3:
      _InputDims = nvinfer1::Dims3 ((int) _info->dimension[2],
          (int) _info->dimension[1], (int) _info->dimension[0]);
      break;

    case 4:
      _InputDims = nvinfer1::Dims4 ((int) _info->dimension[3], (int) _info->dimension[2],
          (int) _info->dimension[1], (int) _info->dimension[0]);
      break;

    default:
      ml_loge ("TensorRT filter does not support %u dimension.", input_rank);
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
