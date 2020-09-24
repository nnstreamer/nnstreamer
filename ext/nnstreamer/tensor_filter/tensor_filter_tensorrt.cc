/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter, TensorRT Module
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 * Copyright (C) 2020 Sangjung Woo <sangjung.woo@samsung.com>
 */
/**
 * @file   tensor_filter_tensorrt.cc
 * @date   24 Sep 2020
 * @brief  Mediapipe module for tensor_filter gstreamer plugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Dongju Chae <dongju.chae@samsung.com>
 *         Sangjung Woo <sangjung.woo@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (TensorRT) for tensor_filter.
 *
 * @note Only support UFF (universal framework format) file as inference model.
 * @note When building the pipeline, cumtom parameter 'input_rank' is required, which is matched to UFF model file.
 */

#include <glib.h>
#include <string.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>

#include <iostream>
#include <memory>
#include <vector>

#include <NvInfer.h>
#include <NvUffParser.h>
#include <cuda_runtime_api.h>

using Severity = nvinfer1::ILogger::Severity;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

/** @brief a global object of ILogger */
class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg) override
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

/** @brief unique ptr deleter */
struct InferDeleter
{
  template <typename T>
  void operator()(T* obj) const
  {
    if (obj)
      obj->destroy();
  }
};

/**
 * @brief class definition for TensorRT custom filter
 */
class TensorRTCore
{
  template <typename T>
  using UniquePtr = std::unique_ptr<T, InferDeleter>;

public:
  TensorRTCore (const char * uff_path);
  ~TensorRTCore ();

  int initTensorInfo(const GstTensorFilterProperties * prop);
  int checkUnifiedMemory();

  int buildEngine();
  int infer(const GstTensorMemory * inputData, GstTensorMemory * outputData);

  int getInputTensorDim(GstTensorsInfo * info);
  int getOutputTensorDim(GstTensorsInfo * info);

  const char * getUFFModelPath();

private:
  char * _uff_path;
  void * _inputBuffer;

  GstTensorsInfo _inputTensorMeta;
  GstTensorsInfo _outputTensorMeta;

  nvinfer1::Dims _InputDims;
  nvinfer1::Dims _OutputDims;
  nvinfer1::DataType _DataType;

  guint64 _input_rank;

  UniquePtr<nvinfer1::ICudaEngine> _Engine{nullptr};
  UniquePtr<nvinfer1::IExecutionContext> _Context{nullptr};

  int allocBuffer (void **buffer, gsize size);
  int setTensorType(tensor_type t);
  void parseCustomOption (const GstTensorFilterProperties * prop);

  template <typename T>
  UniquePtr<T> makeUnique(T* t)
  {
    return UniquePtr<T>{t};
  }
};

void init_filter_tensorrt (void) __attribute__ ((constructor));
void fini_filter_tensorrt (void) __attribute__ ((destructor));

/**
 * @brief constructor of TensorRTCore
 * @param[uff_path] UFF file path
 * @return Nothing
 */
TensorRTCore::TensorRTCore (const char * uff_path)
{
  _uff_path = g_strdup (uff_path);
  _inputBuffer = nullptr;

  gst_tensors_info_init (&_inputTensorMeta);
  gst_tensors_info_init (&_outputTensorMeta);

  /* Set as invalid dimension value */
  _input_rank = 0;
}

/**
 * @brief destructor of TensorRTCore
 * @return	Nothing
 */
TensorRTCore::~TensorRTCore ()
{
  gst_tensors_info_free (&_inputTensorMeta);
  gst_tensors_info_free (&_outputTensorMeta);

  /* cleanup Cuda memory for input */
  if (_inputBuffer != nullptr)
    cudaFree (_inputBuffer);

  g_free (_uff_path);
}

/**
 * @brief allocate a GPU buffer memory
 * @param[out] buffer : pointer to allocated memory
 * @param[in] size : allocation size in bytes
 * @return 0 if OK. non-zero if error.
 */
int
TensorRTCore::allocBuffer (void **buffer, gsize size)
{
  cudaError_t status = cudaMallocManaged (buffer, size);

  if (status != cudaSuccess) {
    ml_loge ("Failed to allocate Cuda memory");
    return -1;
  }

  return 0;
}

/**
 * @brief return whether Unified Memory is supported or not.
 * @return 0 if Unified Memory is supported. non-zero if error.
 * @note After Cuda version 6, logical Unified Memory is supported in programming language level.
 * However, if the target device is not supported, then cudaMemcpy() internally occurs
 * and it makes performance degradation.
 */
int
TensorRTCore::checkUnifiedMemory()
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
 * @brief return wehter Unified Memory is supported or not.
 * @param[in] t : data type of NNStreamer
 * @return 0 if OK. non-zero if error.
 * @note TensorRT supports kFLOAT(32bit), kHALF(16bit), kINT8, kINT32 and kBOOL.
 * However, NNStreamer does not support kHALF and kBOOL so error returns in this case.
 */
int
TensorRTCore::setTensorType(tensor_type t)
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
 * @brief return parse the rank of input tensor
 */
void
TensorRTCore::parseCustomOption(const GstTensorFilterProperties * prop)
{
  if (prop->custom_properties) {
    gchar **options;
    guint len;

    options = g_strsplit (prop->custom_properties, ",", -1);
    len = g_strv_length (options);

    for (guint i=0; i<len; ++i) {
      gchar **option = g_strsplit (options[i], ":", -1);
      if (g_strv_length (option) > 1) {
        g_strstrip (option[0]);
        g_strstrip (option[1]);

        if (g_ascii_strcasecmp (option[0], "input_rank") == 0) {
          _input_rank = g_ascii_strtoull (option[1], nullptr, 10);
        }
      }
      g_strfreev (option);  
    }
    g_strfreev (options);
  }
}

/**
 * @brief return UFF model file path
 * @return UFF model file path.
 */
const char *
TensorRTCore::getUFFModelPath()
{
  return _uff_path;
}

/**
 * @brief initialize internal data structure based on TensorInfo
 * @param[in] prop : property of tensor_filter instance
 * @return 0 if OK. non-zero if error.
 */
int
TensorRTCore::initTensorInfo(const GstTensorFilterProperties * prop)
{ 
  gst_tensors_info_copy (&_inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&_outputTensorMeta, &prop->output_meta);

  /* Set Custom property */
  parseCustomOption(prop);

  /*
   * TensorRT supports 2, 3 and 4 ranks of a tensor.
   */
  switch (_input_rank) {
    case 2:
      _InputDims = nvinfer1::Dims2(
        (int) _inputTensorMeta.info[0].dimension[1],
        (int) _inputTensorMeta.info[0].dimension[0]);
      break;

    case 3:
    _InputDims = nvinfer1::Dims3(
      (int) _inputTensorMeta.info[0].dimension[2],
      (int) _inputTensorMeta.info[0].dimension[1],
      (int) _inputTensorMeta.info[0].dimension[0]);
      break;
    
    case 4:
    _InputDims = nvinfer1::Dims4(
      (int) _inputTensorMeta.info[0].dimension[3],
      (int) _inputTensorMeta.info[0].dimension[2],
      (int) _inputTensorMeta.info[0].dimension[1],
      (int) _inputTensorMeta.info[0].dimension[0]);
      break;

    default:
      ml_loge ("TensorRT filter does not support %lu dimension.", _input_rank);
      return -1;
  }

  if (setTensorType(_inputTensorMeta.info[0].type) != 0) {
    ml_loge ("TensorRT filter does not support the input data type.");
    return -1;
  }

  return 0;
}

/**
 * @brief build TensorRT Engine and ExecutionContext obejct for inference
 * @return 0 if OK. non-zero if error.
 */ 
int
TensorRTCore::buildEngine()
{
  /* Make builder, network, config, parser object */
  auto builder = makeUnique(nvinfer1::createInferBuilder(gLogger));
  if (!builder) {
    ml_loge ("Failed to create builder");
    return -1;
  }

  auto network = makeUnique(builder->createNetwork());
  if (!network) {
    ml_loge ("Failed to create network");
    return -1;
  }

  auto config = makeUnique(builder->createBuilderConfig());
  if (!config) {
    ml_loge ("Failed to create config");
    return -1;
  }

  auto parser = makeUnique(nvuffparser::createUffParser());
  if (!parser) {
    ml_loge ("Failed to create parser");
    return -1;
  }

  /* Register tensor input & output */
  parser->registerInput(_inputTensorMeta.info[0].name,
    _InputDims, nvuffparser::UffInputOrder::kNCHW);
  parser->registerOutput(_outputTensorMeta.info[0].name);

  /* Parse the imported model */
  parser->parse (_uff_path, *network, _DataType);

  /* Set config */
  builder->setMaxBatchSize(1);
  config->setMaxWorkspaceSize(1 << 20);
  config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

  /* Create Engine object */
  _Engine = makeUnique(builder->buildEngineWithConfig(*network, *config));
  if (!_Engine) {
    ml_loge ("Failed to create the TensorRT Engine object");
    return -1;
  }

  /* Create ExecutionContext obejct */
  _Context = makeUnique(_Engine->createExecutionContext());
  if (!_Context) {
    ml_loge ("Failed to create the TensorRT ExecutionContext object");
    return -1;
  }

  return 0;
}

/**
 * @brief run inference with the given input data
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TensorRTCore::infer(const GstTensorMemory * input,
  GstTensorMemory * output)
{
  /* If internal _inputBuffer is NULL, tne allocate GPU memory */
  if (!_inputBuffer) {
    if (allocBuffer (&_inputBuffer, input->size) != 0) {
      ml_loge ("Failed to allocate GPU memory for input");
      return -1;
    }
  }

  /* Copy input data to Cuda memory space */
  memcpy (_inputBuffer, input->data, input->size);

  /* Allocate output buffer */
  if (allocBuffer (&output->data, output->size) != 0) {
    ml_loge ("Failed to allocate GPU memory for output");
    return -1;
  }

  /* Bind the input and execute the network */
  std::vector<void*> bindings = {_inputBuffer, output->data};
  if (!_Context->execute(1, bindings.data())) {
    ml_loge ("Failed to execute the network");
    return -1;
  }

  /* wait for GPU to finish the inference */
  cudaDeviceSynchronize();

  return 0;
}

/**
 * @brief get input tensors information
 * @param[out] info The dimesions and types of input tensors
 * @return 0 if OK
 */
int
TensorRTCore::getInputTensorDim(GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &_inputTensorMeta);
  return 0;
}

/**
 * @brief get output tensors information
 * @param[out] info The dimesions and types of output tensors
 * @return 0 if OK
 */
int
TensorRTCore::getOutputTensorDim(GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &_outputTensorMeta);
  return 0;
}

/**
 * @brief  tensor_filter callback to close
 */ 
static void
tensorrt_close (const GstTensorFilterProperties * prop, void **private_data)
{
  TensorRTCore *core = static_cast<TensorRTCore *> (*private_data);
  if (!core)
    return;

  delete core;
  *private_data = nullptr;
}

/**
 * @brief load UFF model file and make internal object for TensorRT inference
 * @param[in] prop : property of tensor_filter instance
 * @param[out] private_data : private data of TensorRT plugin
 * @return 0 if successfully loaded.
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed.
 */ 
static int
tensorrt_loadModel (const GstTensorFilterProperties * prop, void **private_data)
{
  TensorRTCore * core;
  const gchar * uff_file;

  if (prop->num_models != 1) {
    ml_loge ("TensorRT filter requires one UFF model file\n");
    return -1;
  }

  uff_file = prop->model_files[0];
  if (uff_file == nullptr) {
    ml_loge ("UFF model file is not valid\n");
    return -1;
  }

  /* check existing core*/
  core = static_cast<TensorRTCore *>(*private_data);
  if (core != nullptr) {
    if (g_strcmp0 (uff_file, core->getUFFModelPath()) == 0) {
      /* Skip */
      return 0;
    }
    tensorrt_close (prop, private_data);
    core = nullptr;
  }

  /* TODO use try/catch block */
  core = new TensorRTCore(uff_file);
  if (core == nullptr) {
    ml_loge ("Failed to allocate memory for filter subplugin: TensorRT\n");
    return -1;
  }

  if (core->checkUnifiedMemory() != 0) {
    ml_loge ("Failed to enable unified memory");
    goto error;
  }

  if (core->initTensorInfo(prop) != 0) {
    ml_loge ("Failed to initialize an object: TensorRT\n");
    goto error;
  }

  if (core->buildEngine() != 0) {
    ml_loge ("Failed to build a TensorRT engine");
    goto error;
  }

  *private_data = core;
  return 0;

error:
  delete core;
  return -2;  
}

/**
 * @brief  tensor_filter callback to open
 */ 
static int
tensorrt_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int status = tensorrt_loadModel (prop, private_data);

  return status;
}

/**
 * @brief invoke_NN callback for GstTensorFilterFramework
 * @param[in] prop : property of tensor_filter instance
 * @param[in] private_data : TensorRT plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error
 */
static int
tensorrt_infer (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  TensorRTCore *core = static_cast<TensorRTCore *> (*private_data);
  g_return_val_if_fail (core && input && output, -EINVAL);

  return core->infer (input, output);
}

/**
 * @brief getInputDimension callback for GstTensorFilterFramework
 * @param[in] prop ; property of tensor_filter instance
 * @param[in] private_data : TensorRT plugin's private data
 * @param[out] info The dimesions and types of input tensors
 * @return 0
 */
static int
tensorrt_getInputDim (const GstTensorFilterProperties * prop,
  void **private_data, GstTensorsInfo * info)
{
  TensorRTCore *core = static_cast<TensorRTCore *> (*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getInputTensorDim (info);
}

/**
 * @brief getOutputDimension callback for GstTensorFilterFramework
 * @param[in] prop ; property of tensor_filter instance
 * @param[in] private_data : TensorRT plugin's private data
 * @param[out] info The dimesions and types of output tensors
 * @return 0
 */
static int
tensorrt_getOutputDim (const GstTensorFilterProperties * prop,
  void **private_data, GstTensorsInfo * info)
{
  TensorRTCore *core = static_cast<TensorRTCore *> (*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getOutputTensorDim (info);
}

/**
 * @brief The destroyNotify callback for GstTensorFilterFramework
 * @param[in] private_data : TensorRT plugin's private data
 * @param[in] data : Cuda data buffer
 * @return None
 */
static void
tensorrt_destroyNotify (void **private_data, void *data)
{
  if (data != nullptr)
    cudaFree (data);
}

static int
tensorrt_checkAvailability (accl_hw hw)
{
  /* TODO add HW acceleration enum */
  return 0;
}

static gchar filter_subplugin_tensorrt[] = "tensorrt";

static GstTensorFilterFramework NNS_support_tensorrt = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = tensorrt_open,
  .close = tensorrt_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_tensorrt (void)
{
  NNS_support_tensorrt.name = filter_subplugin_tensorrt;
  NNS_support_tensorrt.allow_in_place = FALSE;
  NNS_support_tensorrt.allocate_in_invoke = TRUE;
  NNS_support_tensorrt.run_without_model = FALSE;
  NNS_support_tensorrt.verify_model_path = FALSE;
  NNS_support_tensorrt.invoke_NN = tensorrt_infer;
  NNS_support_tensorrt.getInputDimension = tensorrt_getInputDim;
  NNS_support_tensorrt.getOutputDimension = tensorrt_getOutputDim;
  NNS_support_tensorrt.destroyNotify = tensorrt_destroyNotify;
  NNS_support_tensorrt.checkAvailability = tensorrt_checkAvailability;

  nnstreamer_filter_probe (&NNS_support_tensorrt);
}

/** @brief Destruct the subplugin */
void
fini_filter_tensorrt (void)
{
  nnstreamer_filter_exit (NNS_support_tensorrt.name);
}
