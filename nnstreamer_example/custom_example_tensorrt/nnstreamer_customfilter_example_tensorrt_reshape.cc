// SPDX-License-Identifier: LGPL-2.1-only

/**
 * NNStreamer TensorRT Custom Filter Example: Reshape
 *
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 *
 * @file  nnstreamer_customfilter_example_tensorrt_reshape.cc
 * @date  14 Sep 2020
 * @brief  TensorRT Custom NNStreamer Filter Example: Reshape
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug  No known bugs
 *
 * Reference code: sampleDynamicReshape.cpp in TensorRT samples
 *
 * This example reshapes the inputs to the given dimensions.
 * The custom property is to be given as, "custom=D1:D2:D3"
 * E.g., custom=224:224:3
 */

#include <glib.h>
#include <string.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api.h>

#include <iostream>
#include <memory>
#include <vector>

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>

using Severity = nvinfer1::ILogger::Severity;

/** @brief a global object of ILogger */
class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char* msg) override
  {
    switch (severity) {
      case Severity::kWARNING:
        g_warning ("%s", msg);
        break;
      case Severity::kINFO:
        g_message ("%s", msg);
        break;
      case Severity::kVERBOSE:
        g_debug ("%s", msg);
        break;
      default:
        g_critical ("%s", msg);
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
class CustomTensorRT
{
  template <typename T>
  using UniquePtr = std::unique_ptr<T, InferDeleter>;

  public:
    CustomTensorRT ();
    ~CustomTensorRT ();

    gboolean buildEngine ();
    gboolean checkUnifiedMemory ();

    gboolean setInputMeta (const GstTensorsInfo *info);
    gboolean setOutputMeta (const GstTensorsInfo *info);

    /** @brief get input tensors info */
    const GstTensorsInfo * getInputMeta() { return &mInputMeta; }
    /** @brief get output tensors info */
    const GstTensorsInfo * getOutputMeta() { return &mOutputMeta; }

    gboolean infer (const GstTensorMemory * input, GstTensorMemory * output);

  private:
    GstTensorsInfo mInputMeta;
    GstTensorsInfo mOutputMeta;

    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;

    void * mInput;

    UniquePtr<nvinfer1::ICudaEngine> mEngine{nullptr};
    UniquePtr<nvinfer1::IExecutionContext> mContext{nullptr};

    gboolean allocBuffer (void **buffer, gsize size);
    gboolean resizeInput (const GstTensorsInfo *info);

    /** @brief wrapper to make unique ptr */
    template <typename T>
    UniquePtr<T> makeUnique(T* t)
    {
      return UniquePtr<T>{t};
    }
};

/**
 * @brief constructor of CustomTensorRT
 */
CustomTensorRT::CustomTensorRT ()
{
  gst_tensors_info_init (&mInputMeta);
  gst_tensors_info_init (&mOutputMeta);

  mInput = nullptr;
}

/**
 * @brief destructor of CustomTensorRT
 */
CustomTensorRT::~CustomTensorRT ()
{
  gst_tensors_info_free (&mInputMeta);
  gst_tensors_info_free (&mOutputMeta);

  if (mInput != nullptr)
    cudaFree (&mInput);
}

/**
 * @brief allocate a GPU buffer
 */
gboolean
CustomTensorRT::allocBuffer (void **buffer, gsize size)
{
  cudaError_t status = cudaMallocManaged (buffer, size);

  if (status != cudaSuccess) {
    g_critical ("Failed to allocate GPU memory");
    return false;
  }

  return true;
}

/**
 * @brief resize a GPU buffer for input
 */
gboolean
CustomTensorRT::resizeInput (const GstTensorsInfo *info)
{
  gsize input_size = gst_tensor_info_get_size (&info->info[0]);
  void * buffer;

  if (!allocBuffer (&buffer, input_size))
    return false;

  if (mInput != nullptr)
    cudaFree (&mInput);
  mInput = buffer;

  return true;
}

/**
 * @brief set input metadata if required
 */
gboolean
CustomTensorRT::setInputMeta (const GstTensorsInfo *info)
{
  /* skip if dimension is not changed */
  if (gst_tensors_info_is_equal (info, &mInputMeta))
    return true;

  if (!resizeInput (info))
    return false;

  /* TensorRT uses the NCHW data format */
  mInputDims = nvinfer1::Dims4 {
    (int) info->info[0].dimension[3],
    (int) info->info[0].dimension[2],
    (int) info->info[0].dimension[1],
    (int) info->info[0].dimension[0]
  };

  gst_tensors_info_copy (&mInputMeta, info);

  return true;
}

/**
 * @brief set output metadata
 */
gboolean
CustomTensorRT::setOutputMeta (const GstTensorsInfo *info)
{
  /* TensorRT uses the NCHW data format */
  mOutputDims = nvinfer1::Dims4 {
    (int) info->info[0].dimension[3],
    (int) info->info[0].dimension[2],
    (int) info->info[0].dimension[1],
    (int) info->info[0].dimension[0]
  };

  gst_tensors_info_copy (&mOutputMeta, info);

  return true;
}

/**
 * @brief build TensorRT engine
 */
gboolean
CustomTensorRT::buildEngine()
{
  auto builder = makeUnique(nvinfer1::createInferBuilder(gLogger));
  if (!builder) {
    g_critical ("Failed to create builder");
    return false;
  }

  auto network = makeUnique(
      builder->createNetworkV2(
        1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (!network) {
    g_critical ("Failed to create model");
    return false;
  }

  /* dynamic input shape */
  auto input = network->addInput("input", nvinfer1::DataType::kFLOAT,
      nvinfer1::Dims4{1, -1, -1, -1});
  auto resize = network->addResize(*input);

  resize->setOutputDimensions(mOutputDims);
  network->markOutput(*resize->getOutput(0));

  auto config = makeUnique(builder->createBuilderConfig());
  if (!config) {
    g_critical ("Failed to create builder config");
    return false;
  }

  /* specifies a range of input dimensions */
  auto profile = builder->createOptimizationProfile();
  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
      nvinfer1::Dims4{1,1,1,1});
  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
      nvinfer1::Dims4{1,3,480,640});
  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
      nvinfer1::Dims4{1,16,960,1280});
  config->addOptimizationProfile(profile);

  config->setMaxWorkspaceSize(16 * (1 << 20));
  builder->setMaxBatchSize(1);

  mEngine = makeUnique(builder->buildEngineWithConfig(*network, *config));
  if (!mEngine) {
    g_critical ("Failed to create builder config");
    return false;
  }

  mContext = makeUnique(mEngine->createExecutionContext());
  if (!mContext) {
    g_critical ("Failed to create execution context");
    return false;
  }

  return true;
}

/**
 * @brief run inference with the given input data
 */
gboolean
CustomTensorRT::infer(const GstTensorMemory * input, GstTensorMemory * output)
{
  /**
   * still need to explicitly perform memcpy() for input tensors
   * unless Address Translation Services (ATS) is supported.
   */
  memcpy (mInput, input->data, input->size);

  /* allocate output buffer */
  if (!allocBuffer (&output->data, output->size)) {
    g_critical ("Failed to allocate GPU memory for output");
    return false;
  }

  /* set the input dimensions */
  if (!mContext->setBindingDimensions(0, mInputDims)) {
    g_critical ("Failed to set binding dimensions");
    return false;
  }

  /* all dynamic input dimensions are specified */
  if (!mContext->allInputDimensionsSpecified()) {
    g_critical ("Not all input dimensions are specified");
    return false;
  }

  /* bind the input/output and execute the network */
  std::vector<void*> bindings = {mInput, output->data};
  if (!mContext->executeV2(bindings.data())) {
    g_critical ("Failed to execute the network");
    return false;
  }

  /* wait for GPU to finish the inference */
  cudaDeviceSynchronize();

  return true;
}

/**
 * @brief check and set unified memory capability
 */
gboolean
CustomTensorRT::checkUnifiedMemory()
{
  int version;

  if (cudaRuntimeGetVersion (&version) != cudaSuccess)
    return false;

  /* Unified memory requires at least CUDA-6 */
  if (version < 6000)
    return false;

  return true;
}

/**
 * @brief init callback of tensor_filter custom
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  CustomTensorRT * trt = new CustomTensorRT;
  GstTensorsInfo info;

  gst_tensors_info_init (&info);

  if (prop->custom_properties && strlen (prop->custom_properties) > 0) {
    gchar **strv = g_strsplit (prop->custom_properties, ":", -1);
    gsize i;

    if (g_strv_length (strv) != NNS_TENSOR_RANK_LIMIT - 1) {
      g_critical ("Please specify a proper 'custom' property");
      goto err;
    }

    info.num_tensors = 1;
    for (i = 0; i < NNS_TENSOR_RANK_LIMIT - 1; i++) {
      info.info[0].type = _NNS_FLOAT32;
      info.info[0].dimension[i] = (int) g_ascii_strtoll (strv[i], NULL, 10);
    }
    info.info[0].dimension[NNS_TENSOR_RANK_LIMIT - 1] = 1;

    g_strfreev (strv);
  } else {
    g_critical ("Please specify a 'custom' property");
    goto err;
  }

  if (!trt->checkUnifiedMemory ()) {
    g_critical ("Failed to enable unified memory");
    goto err;
  }

  if (!trt->setOutputMeta (&info)) {
    g_critical ("Failed to allocate output buffer");
    goto err;
  }

  if (!trt->buildEngine ()) {
    g_critical ("Failed to build a TensorRT engine");
    goto err;
  }

  return trt;

err:
  delete trt;
  return nullptr;
}

/**
 * @brief exit callback of tensor_filter custom
 */
static void
pt_exit (void *private_data, const GstTensorFilterProperties * prop)
{
  CustomTensorRT *trt = static_cast<CustomTensorRT *> (private_data);
  g_assert (trt);

  delete trt;
}

/**
 * @brief setInputDimension callback of tensor_filter custom
 */
static int
set_inputDim (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  CustomTensorRT *trt = static_cast<CustomTensorRT *> (private_data);
  g_assert (trt);

  if (!trt->setInputMeta (in_info))
    return -1;

  gst_tensors_info_copy (out_info, trt->getOutputMeta ());

  return 0;
}

/**
 * @brief invoke callback of tensor_filter custom
 */
static int
pt_invoke (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  CustomTensorRT *trt = static_cast<CustomTensorRT *> (private_data);

  if (!trt->infer (input, output))
    return -1;

  return 0;
}

/**
 * @brief destroy notify callback of tensor_filter custom
 */
static void
pt_destroy_notify (void * data)
{
  g_assert (data);
  cudaFree (data);
}

/**
 * @brief tensor_filter custom subplugin definition
 */
static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = pt_init,
  .exitfunc = pt_exit,
  .getInputDim = NULL,
  .getOutputDim = NULL,
  .setInputDim = set_inputDim,
  .invoke = NULL,
  .allocate_invoke = pt_invoke,
  .destroy_notify = pt_destroy_notify,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
