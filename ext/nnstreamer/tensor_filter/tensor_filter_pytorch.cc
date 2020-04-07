/**
 * GStreamer Tensor_Filter, PyTorch Module
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
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
 * @file    tensor_filter_pytorch.cc
 * @date    24 April 2019
 * @brief   PyTorch module for tensor_filter gstreamer plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (pytorch) for tensor_filter.
 *
 */

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

#include <nnstreamer_conf.h>

#include <torch/script.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

#define INPUT_TENSOR_META_CHAR "InputTensorMeta"
#define OUTPUT_TENSOR_META_CHAR "OutputTensorMeta"

static const gchar *torch_accl_support[] = {
  ACCL_AUTO_STR,
  ACCL_DEFAULT_STR,
  ACCL_CPU_STR,
  ACCL_GPU_STR,
  NULL
};

/**
 * @brief	ring cache structure
 */
class TorchCore
{
public:
  TorchCore (const char *_model_path);
  ~TorchCore ();

  int init (const GstTensorFilterProperties * prop);
  int loadModel ();
  const char *getModelPath ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int invoke (const GstTensorMemory * input, GstTensorMemory * output);

private:

  char *model_path;
  bool use_gpu;
  accl_hw accelerator;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */
  bool configured;
  bool first_run;           /**< must be reset after setting input info */

  std::shared_ptr < torch::jit::script::Module > model;

  void setAccelerator (const char *accelerators);
  tensor_type getTensorTypeFromTorch (torch::Dtype torchType);
  bool getTensorTypeToTorch (tensor_type tensorType, torch::Dtype * torchType);
  int validateOutputTensor (at::Tensor output);
  int fillTensorDim (torch::autograd::Variable tensor_meta, tensor_dim dim);
  int processIValue (torch::jit::IValue value, GstTensorMemory * output);
};

void init_filter_torch (void) __attribute__ ((constructor));
void fini_filter_torch (void) __attribute__ ((destructor));

/**
 * @brief	TorchCore creator
 * @param	_model_path	: the logical path to '{model_name}.pth' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TorchCore::TorchCore (const char *_model_path)
{
  g_assert (_model_path != NULL);
  model_path = g_strdup (_model_path);
  configured = false;
  use_gpu = false;
  first_run = true;

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
}

/**
 * @brief	TorchCore Destructor
 * @return	Nothing
 */
TorchCore::~TorchCore ()
{
  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
  g_free (model_path);
}

/**
 * @brief	Set the accelerator for the pytorch
 */
void
TorchCore::setAccelerator (const char *accelerators)
{
  use_gpu = TRUE;
  accelerator = parse_accl_hw (accelerators, torch_accl_support);
  if (accelerator == ACCL_NONE)
    goto use_gpu_ini;
  if ((accelerator & (ACCL_CPU | ACCL_DEFAULT)) != 0)
    use_gpu = FALSE;

  return;

use_gpu_ini:
  use_gpu = nnsconf_get_custom_value_bool ("pytorch", "enable_use_gpu",
      FALSE);
  if (use_gpu == FALSE) {
    accelerator = ACCL_NONE;
  } else {
    accelerator = ACCL_GPU;
  }
}

/**
 * @brief	initialize the object with torch model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 */
int
TorchCore::init (const GstTensorFilterProperties * prop)
{
  setAccelerator (prop->accl_str);
  g_message ("gpu = %d, accl = %s", use_gpu, get_accl_hw_str(accelerator));

  gst_tensors_info_copy (&inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&outputTensorMeta, &prop->output_meta);

  if (loadModel ()) {
    ml_loge ("Failed to load model\n");
    return -1;
  }

  first_run = true;
  return 0;
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char *
TorchCore::getModelPath ()
{
  return model_path;
}

/**
 * @brief	load the torch model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 *        -1 if the modelfile is not valid(or not exist).
 *        -2 if the pt file is not loaded.
 */
int
TorchCore::loadModel ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  if (!g_file_test (model_path, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("the file of model_path (%s) is not valid (not regular).",
        model_path);
    return -1;
  }

  model = torch::jit::load (model_path);
  if (model == nullptr) {
    ml_loge ("Failed to read graph.");
    return -2;
  }

  if (use_gpu) {
    model->to (at::kCUDA);
  }

  /** set the model to evaluation mode */
  model->eval ();

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param torchType	: the defined type of PyTorch
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
TorchCore::getTensorTypeFromTorch (torch::Dtype torchType)
{
  switch (torchType) {
    case torch::kU8:
      return _NNS_UINT8;
    case torch::kI8:
      return _NNS_INT8;
    case torch::kI16:
      return _NNS_INT16;
    case torch::kI32:
      return _NNS_INT32;
    case torch::kI64:
      return _NNS_INT64;
    case torch::kF32:
      return _NNS_FLOAT32;
    case torch::kF64:
      return _NNS_FLOAT64;
    case torch::kF16:
    default:
      break;
  }

  return _NNS_END;
}

/**
 * @brief	return the data type of the tensor
 * @param torchType	: the defined type of PyTorch
 * @return the enum of defined _NNS_TYPE
 */
bool
TorchCore::getTensorTypeToTorch (tensor_type tensorType,
    torch::Dtype * torchType)
{
  switch (tensorType) {
    case _NNS_UINT8:
      *torchType = torch::kU8;
      break;
    case _NNS_INT8:
      *torchType = torch::kI8;
      break;
    case _NNS_INT16:
      *torchType = torch::kI16;
      break;
    case _NNS_INT32:
      *torchType = torch::kI32;
      break;
    case _NNS_INT64:
      *torchType = torch::kI64;
      break;
    case _NNS_FLOAT32:
      *torchType = torch::kF32;
      break;
    case _NNS_FLOAT64:
      *torchType = torch::kF64;
      break;
    default:
      return false;
  }

  return true;
}

/**
 * @brief	check the inserted information about output tensor with model
 * @return 0 if OK. non-zero if error.
 *        -1 if the number of output tensors is not matched.
 *        -2 if the type of output tensors is not matched.
 *        -3 if the dimension of output tensors is not matched.
 */
int
TorchCore::validateOutputTensor (const at::Tensor output)
{
  auto tensor_shape = output.sizes ();

  if (tensor_shape[0] != 0 && outputTensorMeta.num_tensors != tensor_shape[0]) {
    ml_loge ("Invalid output meta: different size");
    return -1;
  }

  if (tensor_shape[0] == 0) {
    tensor_type otype = getTensorTypeFromTorch (output.scalar_type ());
    if (outputTensorMeta.info[0].type != otype) {
      ml_loge ("Invalid output meta: different type");
      return -2;
    }
    goto done;
  }

  for (uint i = 0; i < outputTensorMeta.num_tensors; ++i) {
    tensor_type otype;
    gsize num_gst_tensor, num_torch_tensor;
    at::Tensor sliced_output = output.slice (0);
    c10::IntArrayRef sliced_output_sizes;

    otype = getTensorTypeFromTorch (sliced_output.scalar_type ());
    num_gst_tensor =
        gst_tensor_get_element_count (outputTensorMeta.info[i].dimension);

    num_torch_tensor = 1;
    sliced_output_sizes = sliced_output.sizes ();
    for (int j = 0; j < sliced_output.ndimension (); j++) {
      num_torch_tensor *= sliced_output_sizes[j];
    }

    if (outputTensorMeta.info[i].type != otype) {
      ml_loge ("Invalid output meta: different type");
      return -2;
    }

    if (num_gst_tensor != num_torch_tensor) {
      ml_loge ("Invalid output meta: different element size");
      return -3;
    }
  }

done:
  configured = true;
  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
TorchCore::getInputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &inputTensorMeta);
  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
TorchCore::getOutputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &outputTensorMeta);
  return 0;
}

/**
 * @brief	process the IValue after forward and extract data from ivalue.
 * @param[in] value IValue containing the output in tensor form
 * @param[out]  output Output tensor memory
 * @return 0 if OK. non-zero if error.
 *         -1 if output tensor validation fails.
 */
int
TorchCore::processIValue (torch::jit::IValue value, GstTensorMemory * output)
{
  g_assert (value.isTensor ());
  at::Tensor output_tensor = value.toTensor ();

  /** bring from gpu to cpu */
  if (use_gpu) {
    output_tensor.to (at::kCPU);
  }
  /** make the memory contiguous for direct access */
  output_tensor = output_tensor.contiguous ();

  output->type = getTensorTypeFromTorch (output_tensor.scalar_type ());

  /* validate output tensor once */
  if (!configured && validateOutputTensor (output_tensor)) {
    ml_loge ("Output Tensor Information is not valid");
    return -1;
  }

  /** @todo avoid this memcpy */
  std::memcpy (output->data, output_tensor.data_ptr (),
      output_tensor.nbytes ());
  return 0;
}

/**
 * @brief	invoke the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 *         -1 if the input properties are incompatible.
 *         -2 if the output properties are different with model.
 *         -3 if the output is neither a list nor a tensor.
 *         -4 if running the model failed.
 */
int
TorchCore::invoke (const GstTensorMemory * input, GstTensorMemory * output)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  std::vector < torch::jit::IValue > input_feeds;
  torch::jit::IValue output_value;
  torch::Dtype type;
  at::Tensor tensor;

  for (uint i = 0; i < inputTensorMeta.num_tensors; ++i) {
    std::vector < int64_t > input_shape;
    input_shape.assign (&inputTensorMeta.info[i].dimension[0],
        &inputTensorMeta.info[i].dimension[0] + NNS_TENSOR_RANK_LIMIT);

    if (!getTensorTypeToTorch (input[i].type, &type)) {
      ml_loge ("This data type is not valid: %d", input[i].type);
      return -1;
    }
    at::TensorOptions options = torch::TensorOptions ().dtype (type);

    std::reverse (input_shape.begin (), input_shape.end ());
    tensor = torch::from_blob (input[i].data, input_shape, options);

    if (use_gpu) {
      tensor.to (at::kCUDA);
    }

    input_feeds.emplace_back (tensor);
  }

  /**
   * As the input information has not been verified, the first run for the model
   * is encapsulated in a try-catch block
   */
  if (first_run) {
    try {
      output_value = model->forward (input_feeds);
      first_run = false;
    }
    catch (const std::runtime_error & re)
    {
      ml_loge ("Runtime error while running the model: %s", re.what ());
      return -4;
    }
    catch (const std::exception & ex) {
      ml_loge ("Exception while running the model : %s", ex.what ());
      return -4;
    }
    catch (...) {
      ml_loge ("Unknown exception while running the model");
      return -4;
    }
  } else {
    output_value = model->forward (input_feeds);
  }

  if (output_value.isTensor ()) {
    g_assert (outputTensorMeta.num_tensors == 1);
    if (processIValue (output_value, &output[0])) {
      ml_loge ("Output Tensor Information is not valid");
      return -2;
    }
  } else if (output_value.isGenericList ()) {
    std::vector < torch::jit::IValue > output_list =
        output_value.toGenericListRef ();
    g_assert (outputTensorMeta.num_tensors == output_list.size ());
    int idx = 0;
  for (auto & ivalue_element:output_list) {
      if (processIValue (ivalue_element, &output[idx++])) {
        ml_loge ("Output Tensor Information is not valid");
        return -2;
      }
    }
  } else {
    ml_loge ("Output is not a tensor.");
    return -3;
  }


#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Invoke() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  return 0;
}

/**
 * @brief	fill tensor dimension
 * @param tensor_meta	pytorch tensor variable
 * @param[out] dim the array of the tensor dimension
 * @return 0 if OK. non-zero if error.
 */
int
TorchCore::fillTensorDim (torch::autograd::Variable tensor_meta, tensor_dim dim)
{
  int num_dim = tensor_meta.ndimension ();
  g_assert (num_dim <= NNS_TENSOR_RANK_LIMIT);
  /** the order of dimension is reversed at CAPS negotiation */
  std::reverse_copy (tensor_meta.sizes ().begin (),
      tensor_meta.sizes ().end (), dim);

  /** fill the remnants with 1 */
  for (int idx = num_dim; idx < NNS_TENSOR_RANK_LIMIT; ++idx) {
    dim[idx] = 1;
  }

  return 0;
}

/**
 * @brief Free privateData and move on.
 */
static void
torch_close (const GstTensorFilterProperties * prop, void **private_data)
{
  TorchCore *core = static_cast < TorchCore * >(*private_data);

  if (!core)
    return;

  delete core;

  *private_data = NULL;
}

/**
 * @brief Load pytorch modelfile
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static gint
torch_loadModelFile (const GstTensorFilterProperties * prop,
    void **private_data)
{
  TorchCore *core;
  const gchar *model_path;

  if (prop->num_models != 1)
    return -1;

  core = static_cast < TorchCore * >(*private_data);
  model_path = prop->model_files[0];

  if (core != NULL) {
    if (g_strcmp0 (model_path, core->getModelPath ()) == 0)
      return 1;                 /* skipped */

    torch_close (prop, private_data);
  }

  core = new TorchCore (model_path);
  if (core == NULL) {
    g_printerr ("Failed to allocate memory for filter subplugin: PyTorch\n");
    return -1;
  }

  if (core->init (prop) != 0) {
    *private_data = NULL;
    delete core;

    g_printerr ("failed to initialize the object: PyTorch\n");
    return -2;
  }

  *private_data = core;

  return 0;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 */
static gint
torch_open (const GstTensorFilterProperties * prop, void **private_data)
{
  gint status = torch_loadModelFile (prop, private_data);

  return status;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static gint
torch_invoke (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  TorchCore *core = static_cast < TorchCore * >(*private_data);
  g_return_val_if_fail (core && input && output, -EINVAL);

  return core->invoke (input, output);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static gint
torch_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  TorchCore *core = static_cast < TorchCore * >(*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getInputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static gint
torch_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  TorchCore *core = static_cast < TorchCore * >(*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getOutputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] hw backend accelerator hardware
 * @return 0 if supported. -errno if not supported.
 */
static int
torch_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (torch_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}

static gchar filter_subplugin_pytorch[] = "pytorch";

static GstTensorFilterFramework NNS_support_pytorch = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = torch_open,
  .close = torch_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_torch (void)
{
  NNS_support_pytorch.name = filter_subplugin_pytorch;
  NNS_support_pytorch.allow_in_place = FALSE;
  NNS_support_pytorch.allocate_in_invoke = FALSE;
  NNS_support_pytorch.run_without_model = FALSE;
  NNS_support_pytorch.verify_model_path = FALSE;
  NNS_support_pytorch.invoke_NN = torch_invoke;
  NNS_support_pytorch.getInputDimension = torch_getInputDim;
  NNS_support_pytorch.getOutputDimension = torch_getOutputDim;
  NNS_support_pytorch.checkAvailability = torch_checkAvailability;

  nnstreamer_filter_probe (&NNS_support_pytorch);
}

/** @brief Destruct the subplugin */
void
fini_filter_torch (void)
{
  nnstreamer_filter_exit (NNS_support_pytorch.name);
}
