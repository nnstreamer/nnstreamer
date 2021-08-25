/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	tensor_filter_armnn.cc
 * @date	20 Nov 2019
 * @brief	ARM NN module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (armnn) for tensor_filter.
 */

#include <algorithm>
#include <errno.h>
#include <glib.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#include <memory>
#include <map>
#include <vector>

#include <armnn/ArmNN.hpp>

#if ENABLE_ARMNN_CAFFE
#include <armnnCaffeParser/ICaffeParser.hpp>
#endif
#if ENABLE_ARMNN_TFLITE
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#endif

#include <nnstreamer_log.h>
#define NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_filter.h>
#undef NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_conf.h>
#include <nnstreamer_plugin_api.h>

static const gchar *armnn_accl_support[]
    = { ACCL_CPU_NEON_STR, /** ACCL for default and auto config */
        ACCL_CPU_STR, ACCL_GPU_STR, NULL };

/**
 * @brief	ring cache structure
 */
class ArmNNCore
{
  public:
  ArmNNCore (const char *_model_path, accl_hw hw);
  ~ArmNNCore ();

  int init (const GstTensorFilterProperties *prop);
  int loadModel (const GstTensorFilterProperties *prop);
  const char *getModelPath ();
  int setInputTensorProp ();
  int setOutputTensorProp ();
  int getInputTensorDim (GstTensorsInfo *info);
  int getOutputTensorDim (GstTensorsInfo *info);
  int invoke (const GstTensorMemory *input, GstTensorMemory *output);

  private:
  char *model_path;
  accl_hw accel;

  GstTensorsInfo inputTensorMeta; /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta; /**< The tensor info of output tensors */

  armnn::IRuntimePtr runtime;
  armnn::INetworkPtr network;
  armnn::NetworkId networkIdentifier;
  armnn::IRuntime::CreationOptions options;
  std::vector<armnn::BindingPointInfo> inputBindingInfo;
  std::vector<armnn::BindingPointInfo> outputBindingInfo;

  int makeCaffeNetwork (std::map<std::string, armnn::TensorShape> &input_map,
      std::vector<std::string> &output_vec);
  int makeTfLiteNetwork ();
  int makeTfNetwork (std::map<std::string, armnn::TensorShape> &input_map,
      std::vector<std::string> &output_vec);
  int makeNetwork (const GstTensorFilterProperties *prop);

  int setTensorProp (const std::vector<armnn::BindingPointInfo> &bindings,
      GstTensorsInfo *tensorMeta);
  tensor_type getGstTensorType (armnn::DataType armType);
  int getTensorDim (int tensor_idx, tensor_dim dim);
  armnn::Compute getBackend (const accl_hw hw);
};

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_filter_armnn (void) __attribute__((constructor));
void fini_filter_armnn (void) __attribute__((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief	ArmNNCore creator
 * @param	_model_path	: the logical path to model file
 * @param	hw	: hardware accelerator to be used at backend
 */
ArmNNCore::ArmNNCore (const char *_model_path, accl_hw hw)
    : accel (hw), runtime (nullptr, &armnn::IRuntime::Destroy),
      network (armnn::INetworkPtr (nullptr, nullptr))
{
  model_path = g_strdup (_model_path);

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
  networkIdentifier = 0;
}

/**
 * @brief	ArmNNCore Destructor
 */
ArmNNCore::~ArmNNCore ()
{
  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
}

/**
 * @brief	initialize the object with armnn model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initializat<= NNS_TENSOR_SIZE_LIMIT of output tensor is failed.
 */
int
ArmNNCore::init (const GstTensorFilterProperties *prop)
{
  int err;
  if ((err = loadModel (prop))) {
    ml_loge ("Failed to load model\n");
    return err;
  }

  if ((err = setInputTensorProp ())) {
    ml_loge ("Failed to initialize input tensor\n");
    return err;
  }

  if ((err = setOutputTensorProp ())) {
    ml_loge ("Failed to initialize output tensor\n");
    return err;
  }
  return 0;
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char *
ArmNNCore::getModelPath ()
{
  return model_path;
}

#if ENABLE_ARMNN_CAFFE
/**
 * @brief make network with caffe parser
 * @param[in] input_map input data map
 * @param[in] output_vec output data vector
 * @return 0 on success, -errno on error
 */
int
ArmNNCore::makeCaffeNetwork (std::map<std::string, armnn::TensorShape> &input_map,
    std::vector<std::string> &output_vec)
{
  bool unknown_input_dim = false;

  armnnCaffeParser::ICaffeParserPtr parser = armnnCaffeParser::ICaffeParser::Create ();

  for (auto const &inputs : input_map) {
    if (inputs.second.GetNumDimensions () == 0) {
      unknown_input_dim = true;
    }
  }

  if (unknown_input_dim) {
    network = parser->CreateNetworkFromBinaryFile (model_path, {}, output_vec);
  } else {
    network = parser->CreateNetworkFromBinaryFile (model_path, input_map, output_vec);
  }

  /** set input/output bindings */
  for (auto const &output_name : output_vec) {
    outputBindingInfo.push_back (parser->GetNetworkOutputBindingInfo (output_name));
  }

  for (auto const &inputs : input_map) {
    inputBindingInfo.push_back (parser->GetNetworkInputBindingInfo (inputs.first));
  }

  return 0;
}
#else /* ENABLE_ARMNN_CAFFE */
/**
 * @brief A dummy Caffe parser used when Caffe was not enabled at build time.
 * @retval -EPERM this returns permission error always.
 */
int
ArmNNCore::makeCaffeNetwork (std::map<std::string, armnn::TensorShape> &input_map,
    std::vector<std::string> &output_vec)
{
  g_printerr ("ARMNN-CAFFE was not enabled at build-time. tensor-filter::armnn cannot handle caffe networks.");
  return -EPERM;
}
#endif

/**
 * @brief make network with tensorflow parser
 * @param[in] input_map input data map
 * @param[in] output_vec output data vector
 * @return 0 on success, -errno on error
 */
int
ArmNNCore::makeTfNetwork (std::map<std::string, armnn::TensorShape> &input_map,
    std::vector<std::string> &output_vec)
{
  /** @todo fill this */
  return -EPERM;
}

#if ENABLE_ARMNN_TFLITE
/**
 * @brief make network with tensorflow-lite parser
 * @return 0 on success, -errno on error
 */
int
ArmNNCore::makeTfLiteNetwork ()
{
  /** Tensorflow-lite parser */
  armnnTfLiteParser::ITfLiteParserPtr parser
      = armnnTfLiteParser::ITfLiteParser::Create ();
  if (!parser)
    return -EINVAL;

  network = parser->CreateNetworkFromBinaryFile (model_path);
  if (!network)
    return -EPERM;

  /** @todo: support multiple subgraphs */
  /** set input/output bindings */
  std::vector<std::string> in_names = parser->GetSubgraphInputTensorNames (0);
  for (auto const &name : in_names) {
    inputBindingInfo.push_back (parser->GetNetworkInputBindingInfo (0, name));
  }

  std::vector<std::string> out_names = parser->GetSubgraphOutputTensorNames (0);
  for (auto const &name : out_names) {
    outputBindingInfo.push_back (parser->GetNetworkOutputBindingInfo (0, name));
  }
  return 0;
}
#else /* ENABLE_ARMNN_TFLITE */
/**
 * @brief A dummy function used when tensorflow-lite is not enabled at build-time.
 * @retval -EPERM this returns error always.
 */
int
ArmNNCore::makeTfLiteNetwork ()
{
  return -EPERM;
}
#endif

/**
 * @brief make network based on the model file received
 * @param[in] prop tensor filter based properties
 * @return 0 on success, -errno on error
 */
int
ArmNNCore::makeNetwork (const GstTensorFilterProperties *prop)
{
  std::vector<std::string> output_vec;
  std::map<std::string, armnn::TensorShape> input_map;

  if (g_str_has_suffix (model_path, ".tflite")) {
    return makeTfLiteNetwork ();
  }

  /** Create output vector with name of the layer */
  if (prop->output_meta.num_tensors != 0) {
    output_vec.reserve (prop->output_meta.num_tensors);
    for (unsigned int i = 0; i < prop->output_meta.num_tensors; i++) {
      if (prop->output_meta.info[i].name == NULL) {
        /** clear output vec in case of error */
        output_vec.clear ();
        output_vec.shrink_to_fit ();
        break;
      }
      output_vec.push_back (prop->output_meta.info[i].name);
    }
  }

  /** Create input map with name and data shape */
  for (unsigned int i = 0; i < prop->input_meta.num_tensors; i++) {
    if (prop->input_meta.info[i].name == NULL) {
      /** clear input map in case of error */
      input_map.clear ();
      break;
    }

    /** Set dimension only if valid */
    if (gst_tensor_dimension_is_valid (prop->input_meta.info[i].dimension)) {
      unsigned int rev_dim[NNS_TENSOR_RANK_LIMIT];
      std::reverse_copy (prop->input_meta.info[i].dimension,
          prop->input_meta.info[i].dimension + NNS_TENSOR_RANK_LIMIT, rev_dim);
      input_map[prop->input_meta.info[i].name]
          = armnn::TensorShape (NNS_TENSOR_RANK_LIMIT, rev_dim);
    } else {
      input_map[prop->input_meta.info[i].name] = armnn::TensorShape ();
    }
  }

  if (g_str_has_suffix (model_path, ".prototxt") || g_str_has_suffix (model_path, ".pb")) {
    return makeTfNetwork (input_map, output_vec);
  } else if (g_str_has_suffix (model_path, ".caffemodel")) {
    return makeCaffeNetwork (input_map, output_vec);
  }

  return -EINVAL;
}

/**
 * @brief ARMNN backend retrieval
 */
armnn::Compute
ArmNNCore::getBackend (const accl_hw hw)
{
  switch (hw) {
  case ACCL_GPU:
    return armnn::Compute::GpuAcc;
  case ACCL_NONE:
  /** intended */
  case ACCL_CPU:
    return armnn::Compute::CpuRef;
  case ACCL_CPU_NEON:
  /** intended */
  default:
    return armnn::Compute::CpuAcc;
  }
}

/**
 * @brief	load the armnn model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
ArmNNCore::loadModel (const GstTensorFilterProperties *prop)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  std::vector<std::string> output_vec;
  std::map<std::string, armnn::TensorShape> input_map;
  int err = 0;
  armnn::Status status;

  if (!g_file_test (model_path, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("the file of model_path (%s) is not valid (not regular)\n", model_path);
    return -EINVAL;
  }

  try {
    /** Make the network */
    if ((err = makeNetwork (prop)) != 0)
      throw std::runtime_error ("Error in building the network.");

    /* Optimize the network for the given runtime */
    std::vector<armnn::BackendId> backends = { getBackend (accel) };
    /**
     * @todo add option to enable FP32 to FP16 with OptimizerOptions
     * @todo add GPU based optimizations
     * Support these with custom_properties from tensor_filter
     */
    runtime = armnn::IRuntime::Create (options);
    if (!runtime)
      throw std::runtime_error ("Error creating runtime");

    armnn::IOptimizedNetworkPtr optNet
        = armnn::Optimize (*network, backends, runtime->GetDeviceSpec ());
    if (!optNet)
      throw std::runtime_error ("Error optimizing the network.");

    /* Load the network on the device */
    status = runtime->LoadNetwork (networkIdentifier, std::move (optNet));
    if (status == armnn::Status::Failure)
      throw std::runtime_error ("Error loading the network.");
  } catch (...) {
    try {
      runtime = nullptr;
      network = nullptr;
      throw;
    } catch (const std::runtime_error &re) {
      ml_loge ("Runtime error while loading the network: %s", re.what ());
      if (err != 0)
        return err;
      return -EINVAL;
    } catch (const std::exception &ex) {
      ml_loge ("Exception while loading the network : %s", ex.what ());
      return -EINVAL;
    } catch (...) {
      ml_loge ("Unknown exception while loading the network");
      return -EINVAL;
    }
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tfType	: the defined type of Arm NN
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
ArmNNCore::getGstTensorType (armnn::DataType armType)
{
  switch (armType) {
  case armnn::DataType::Signed32:
    /** Supported with tf and tflite */
    return _NNS_INT32;
  case armnn::DataType::Float32:
    /** Supported with tf, tflite and caffe */
    return _NNS_FLOAT32;
  case armnn::DataType::Float16:
    ml_logw ("Unsupported armnn datatype Float16.");
    break;
  case armnn::DataType::QAsymmU8:
    /** Supported with tflite */
    return _NNS_UINT8;
  case armnn::DataType::Boolean:
    ml_logw ("Unsupported armnn datatype Boolean.");
    break;
  case armnn::DataType::QSymmS16:
    ml_logw ("Unsupported armnn datatype QuantisedSym16.");
    break;
  default:
    ml_logw ("Unsupported armnn datatype unknown.");
    /** @todo Support other types */
    break;
  }

  return _NNS_END;
}

/**
 * @brief extract and store the information of given tensor list
 * @param[in] bindings list of tensors in armnn interpreter
 * @param[out] tensorMeta tensors to set the info into
 * @return 0 if OK. non-zero if error.
 */
int
ArmNNCore::setTensorProp (const std::vector<armnn::BindingPointInfo> &bindings,
    GstTensorsInfo *tensorMeta)
{
  if (tensorMeta->num_tensors == 0)
    tensorMeta->num_tensors = bindings.size ();
  else if (tensorMeta->num_tensors != bindings.size ())
    return -EINVAL;

  if (tensorMeta->num_tensors > NNS_TENSOR_SIZE_LIMIT)
    return -EINVAL;

  /** Set using input/output binding info */
  for (unsigned int idx = 0; idx < bindings.size (); ++idx) {
    armnn::TensorInfo arm_info = bindings[idx].second;
    armnn::TensorShape arm_shape;
    GstTensorInfo *gst_info = &tensorMeta->info[idx];

    /* Use binding id as a name, if no name already exists */
    if (gst_info->name == NULL) {
      gst_info->name = g_strdup_printf ("%d", bindings[idx].first);
    }

    /* Set the type */
    if (gst_info->type == _NNS_END) {
      gst_info->type = getGstTensorType (arm_info.GetDataType ());
    } else if (gst_info->type != getGstTensorType (arm_info.GetDataType ())) {
      ml_logw ("Provided data type info does not match with model.");
      return -EINVAL;
    }

    if (gst_info->type == _NNS_END) {
      ml_logw ("Data type not supported.");
      return -EINVAL;
    }

    /* Set the dimensions */
    int num_dim = arm_info.GetNumDimensions ();
    if (num_dim > NNS_TENSOR_RANK_LIMIT) {
      ml_logw ("Data rank exceeds max supported rank.");
      return -EINVAL;
    }

    /** reverse the order of dimensions */
    arm_shape = arm_info.GetShape ();
    for (int i = num_dim - 1; i >= 0; i--) {
      gst_info->dimension[i] = arm_shape[num_dim - i - 1];
    }

    for (int i = NNS_TENSOR_RANK_LIMIT - 1; i >= num_dim; i--) {
      gst_info->dimension[i] = 1;
    }
  }

  return 0;
}

/**
 * @brief extract and store the information of input tensors
 * @return 0 if OK. non-zero if error.
 */
int
ArmNNCore::setInputTensorProp ()
{
  return setTensorProp (inputBindingInfo, &inputTensorMeta);
}

/**
 * @brief extract and store the information of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
ArmNNCore::setOutputTensorProp ()
{
  return setTensorProp (outputBindingInfo, &outputTensorMeta);
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
ArmNNCore::getInputTensorDim (GstTensorsInfo *info)
{
  gst_tensors_info_copy (info, &inputTensorMeta);
  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
ArmNNCore::getOutputTensorDim (GstTensorsInfo *info)
{
  gst_tensors_info_copy (info, &outputTensorMeta);
  return 0;
}

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
ArmNNCore::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  armnn::InputTensors input_tensors;
  armnn::OutputTensors output_tensors;
  armnn::Status ret;

  for (unsigned int i = 0; i < inputTensorMeta.num_tensors; i++) {
    if (inputBindingInfo[i].second.GetNumBytes () != input[i].size) {
      input_tensors.clear ();
      return -EINVAL;
    }
    armnn::ConstTensor input_tensor (inputBindingInfo[i].second, input[i].data);
    input_tensors.push_back ({ inputBindingInfo[i].first, input_tensor });
  }

  for (unsigned int i = 0; i < outputTensorMeta.num_tensors; i++) {
    if (outputBindingInfo[i].second.GetNumBytes () != output[i].size) {
      output_tensors.clear ();
      input_tensors.clear ();
      return -EINVAL;
    }
    armnn::Tensor output_tensor (outputBindingInfo[i].second, output[i].data);
    output_tensors.push_back ({ outputBindingInfo[i].first, output_tensor });
  }

  /** Run the inference */
  ret = runtime->EnqueueWorkload (networkIdentifier, input_tensors, output_tensors);

  /** Clear the Input and Output tensors */
  input_tensors.clear ();
  output_tensors.clear ();

  if (ret == armnn::Status::Failure)
    return -EINVAL;

  return 0;
}

/**
 * @brief Free privateData and move on.
 */
static void
armnn_close (const GstTensorFilterProperties *prop, void **private_data)
{
  ArmNNCore *core;

  core = static_cast<ArmNNCore *> (*private_data);
  if (core == NULL)
    return;

  delete core;
  *private_data = NULL;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : armnn plugin's private data
 */
static int
armnn_open (const GstTensorFilterProperties *prop, void **private_data)
{
  ArmNNCore *core;
  accl_hw hw;
  int err;

  core = static_cast<ArmNNCore *> (*private_data);

  if (core != NULL) {
    if (g_strcmp0 (prop->model_files[0], core->getModelPath ()) != 0) {
      armnn_close (prop, private_data);
    } else {
      return 1;
    }
  }

  if (prop->model_files[0] == NULL)
    return -EINVAL;

  hw = parse_accl_hw (prop->accl_str, armnn_accl_support);
  try {
    core = new ArmNNCore (prop->model_files[0], hw);
  } catch (const std::bad_alloc &ex) {
    g_printerr ("Failed to allocate memory for filter subplugin.");
    return -ENOMEM;
  }

  if ((err = core->init (prop)) != 0) {
    g_printerr ("failed to initialize the object for armnn");
    delete core;
    return err;
  }

  *private_data = core;
  return 0;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : armnn plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
armnn_invoke (const GstTensorFilterProperties *prop, void **private_data,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  ArmNNCore *core;

  g_return_val_if_fail (private_data != NULL, -EINVAL);
  g_return_val_if_fail (*private_data != NULL, -EINVAL);
  g_return_val_if_fail (input != NULL, -EINVAL);
  g_return_val_if_fail (output != NULL, -EINVAL);

  core = static_cast<ArmNNCore *> (*private_data);

  return core->invoke (input, output);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : armnn plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static int
armnn_getInputDim (const GstTensorFilterProperties *prop, void **private_data,
    GstTensorsInfo *info)
{
  ArmNNCore *core;

  g_return_val_if_fail (*private_data != NULL, -EINVAL);
  g_return_val_if_fail (info != NULL, -EINVAL);

  core = static_cast<ArmNNCore *> (*private_data);

  return core->getInputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : armnn plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static int
armnn_getOutputDim (const GstTensorFilterProperties *prop, void **private_data,
    GstTensorsInfo *info)
{
  ArmNNCore *core;

  g_return_val_if_fail (*private_data != NULL, -EINVAL);
  g_return_val_if_fail (info != NULL, -EINVAL);

  core = static_cast<ArmNNCore *> (*private_data);

  return core->getOutputTensorDim (info);
}

/**
 * @brief Check support of the backend
 * @param hw: backend to check support of
 */
static int
armnn_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (armnn_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}

static gchar filter_subplugin_armnn[] = "armnn";

static GstTensorFilterFramework NNS_support_armnn = {.version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = armnn_open,
  .close = armnn_close,
  {.v0 = {
       .name = filter_subplugin_armnn,
       .allow_in_place = FALSE, /** @todo: support this to optimize performance later. */
       .allocate_in_invoke = FALSE,
       .run_without_model = FALSE,
       .verify_model_path = FALSE,
       .statistics = nullptr,
       .invoke_NN = armnn_invoke,
       .getInputDimension = armnn_getInputDim,
       .getOutputDimension = armnn_getOutputDim,
       .setInputDimension = nullptr,
       .destroyNotify = nullptr,
       .reloadModel = nullptr,
       .handleEvent = nullptr,
       .checkAvailability = armnn_checkAvailability,
       .allocateInInvoke = nullptr,
   } } };

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_armnn (void)
{
  nnstreamer_filter_probe (&NNS_support_armnn);
}

/** @brief Destruct the subplugin */
void
fini_filter_armnn (void)
{
  nnstreamer_filter_exit (NNS_support_armnn.v0.name);
}
