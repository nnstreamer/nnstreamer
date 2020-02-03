/**
 * GStreamer Tensor_Filter, Edge-TPU Module
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	tensor_filter_edgetpu.c
 * @date	10 Dec 2019
 * @brief	Edge-TPU module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (Edge TPU) for tensor_filter.
 *
 * @todo This supports single-model-single-TPU only. Prep for multi-TPU.
 * @todo A lot of this duplicate tf-lite filter.
 *       We may be able to embed this code into tf-lite filter code.
 */
#include <iostream>

#include <stdint.h>

#include <nnstreamer_plugin_api_filter.h>
#include <tensor_common.h>
#include <glib.h>

#include <edgetpu.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/builtin_op_data.h>
#include <tensorflow/lite/kernels/register.h>

void init_filter_edgetpu (void) __attribute__ ((constructor));
void fini_filter_edgetpu (void) __attribute__ ((destructor));

/**
 * @brief Internal data structure
 */
typedef struct {
  char *model_path; /**< The model *.tflite file */
  GstTensorsInfo input; /**< Input tensors metadata */
  GstTensorsInfo output; /**< Output tensors metadata */

  /* EdgeTPU + Tensorflow-lite Execution */
  std::unique_ptr<tflite::Interpreter> model_interpreter;
      /**< TFLite interpreter */
  tflite::Interpreter *interpreter;
      /**< model_interpreter.get() */
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
      /**< EdgeTPU Device context */
  std::unique_ptr<tflite::FlatBufferModel> model;
      /**< Loaded TF Lite model (from model_path) */
} pdata;

/**
 * @brief Get TF-Lite interpreter w/ edgetpu context
 */
static std::unique_ptr<tflite::Interpreter>
BuildEdgeTpuInterpreter(const tflite::FlatBufferModel &model,
      edgetpu::EdgeTpuContext* edgetpu_context)
{
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) !=
      kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }

  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }

  return interpreter;
}

static void
edgetpu_close (const GstTensorFilterProperties * prop, void **private_data);

/**
 * @brief Configure private_data
 */
static int
allocateData (const GstTensorFilterProperties * prop, void **private_data)
{
  pdata *data;
  if (*private_data != NULL) {
    /* Already opened */
    data = (pdata *) *private_data;

    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      std::cerr << "Model path is not given." << std::endl;
      return -EINVAL;
    }
    if (data->model_path && g_strcmp0 (prop->model_files[0],
            data->model_path) == 0) {
      return 0; /* Already opened with same file. Skip ops */
    }

    edgetpu_close (prop, private_data); /* Close before opening one. */
  }

  *private_data = data = g_new0 (pdata, 1);
  if (data == NULL) {
    std::cerr << "Failed to allocate memory for edge-tpu tensor_filer."
        << std::endl;
    return -ENOMEM;
  }
  return 0;
}

/**
 * @brief from tflite-core
 * @todo Remove this func or make them shared
 */
static int
getTensorDim (tflite::Interpreter *interpreter, int tensor_idx,
    tensor_dim dim)
{
  TfLiteIntArray *tensor_dims = interpreter->tensor (tensor_idx)->dims;
  int len = tensor_dims->size;
  g_assert (len <= NNS_TENSOR_RANK_LIMIT);

  /* the order of dimension is reversed at CAPS negotiation */
  std::reverse_copy (tensor_dims->data, tensor_dims->data + len, dim);

  /* fill the remnants with 1 */
  for (int i = len; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 1;
  }

  return 0;
}

/**
 * @brief From tflite-core.cc
 * @todo Remove this or make them shared
 */
static tensor_type
getTensorType (TfLiteType tfType)
{
  switch (tfType) {
    case kTfLiteFloat32:
      return _NNS_FLOAT32;
    case kTfLiteUInt8:
      return _NNS_UINT8;
    case kTfLiteInt32:
      return _NNS_INT32;
    case kTfLiteBool:
      return _NNS_INT8;
    case kTfLiteInt64:
      return _NNS_INT64;
    case kTfLiteString:
    default:
      /** @todo Support other types */
      break;
  }

  return _NNS_END;
}

/**
 * @brief extract and store the information of given tensor list
 * @param tensor_idx_list list of index of tensors in tflite interpreter
 * @param[out] tensorMeta tensors to set the info into
 * @return 0 if OK. non-zero if error.
 */
int
setTensorProp (tflite::Interpreter *interpreter,
    const std::vector<int> &tensor_idx_list,
    GstTensorsInfo * tensorMeta)
{
  tensorMeta->num_tensors = tensor_idx_list.size ();
  if (tensorMeta->num_tensors > NNS_TENSOR_SIZE_LIMIT)
    return -EINVAL;

  for (unsigned int i = 0; i < tensorMeta->num_tensors; ++i) {
    if (getTensorDim (interpreter, tensor_idx_list[i], tensorMeta->info[i].dimension)) {
      std::cerr << "failed to get the dimension of tensors" << std::endl;
      return -EINVAL;
    }
    tensorMeta->info[i].type =
        getTensorType (interpreter->tensor (tensor_idx_list[i])->type);
  }
  return 0;
}

/**
 * @brief Standard tensor_filter callback
 */
static int
edgetpu_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int ret = allocateData (prop, private_data);
  pdata *data = (pdata *) *private_data;
  const std::string model_path = prop->model_files[0];

  if (ret)
    goto err;

  g_free (data->model_path);
  data->model_path = g_strdup (prop->model_files[0]);

  /** Read a model */
  data->model =
    tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (NULL == data->model) {
    std::cerr << "Cannot load the model file: " << model_path << std::endl;
    ret = -EINVAL;
    goto err;
  }

  /** Build an interpreter */
  data->edgetpu_context =
    edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  if (NULL == data->edgetpu_context) {
    std::cerr << "Cannot open edge-TPU device." << std::endl;
    ret = -ENODEV;
    goto err;
  }

  data->model_interpreter =
    BuildEdgeTpuInterpreter(*data->model, data->edgetpu_context.get());
  if (NULL == data->model_interpreter) {
    std::cerr << "Edge-TPU device is opened, but cannot get its interpreter."
        << std::endl;
    ret = -ENODEV;
    goto err;
  }

  data->interpreter = data->model_interpreter.get();

  ret = setTensorProp (data->interpreter, data->interpreter->inputs (),
      &data->input);
  if (ret)
    goto err;

  ret = setTensorProp (data->interpreter, data->interpreter->outputs (),
      &data->output);
  if (ret)
    goto err;

  return 0;
err:
  edgetpu_close (prop, private_data);
  return ret;
}

/**
 * @brief Standard tensor_filter callback
 */
static void
edgetpu_close (const GstTensorFilterProperties * prop,
    void **private_data)
{
  pdata *data = (pdata *) *private_data;

  if (data->model_interpreter) {
    data->model_interpreter = NULL; /* delete unique_ptr */
  }
  if (data->interpreter) {
    data->interpreter = NULL; /* it's already freed with model_interpreter */
  }
  if (data->edgetpu_context) {
    data->edgetpu_context.reset();
    data->edgetpu_context = NULL;
  }
  if (data->model) {
    data->model = NULL; /* delete unique_ptr */
  }

  g_free (data->model_path);
  g_free (*private_data);
  *private_data = NULL;
}

/**
 * @brief Standard tensor_filter callback
 * @details Same with tensor_filter_tensorflow_lite.
 */
static int
edgetpu_invoke (const GstTensorFilterProperties *prop,
    void **private_data, const GstTensorMemory *input,
    GstTensorMemory *output)
{
  pdata *data = (pdata *) *private_data;
  unsigned int i;

  std::vector <int> tensors_idx;
  int tensor_idx;
  TfLiteTensor *tensor_ptr;
  TfLiteStatus status;

  if (!data)
    return -1;
  g_assert (data->interpreter);

  /* Configure inputs */
  for (i = 0; i < data->input.num_tensors; i++) {
    tensor_idx = data->interpreter->inputs ()[i];
    tensor_ptr = data->interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == input[i].size);
    tensor_ptr->data.raw = (char *) input[i].data;
    tensors_idx.push_back (tensor_idx);
  }

  /* Configure outputs */
  for (i = 0; i < data->output.num_tensors; ++i) {
    tensor_idx = data->interpreter->outputs ()[i];
    tensor_ptr = data->interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == output[i].size);
    tensor_ptr->data.raw = (char *) output[i].data;
    tensors_idx.push_back (tensor_idx);
  }

  status = data->interpreter->Invoke ();

  /** if it is not `nullptr`, tensorflow makes `free()` the memory itself. */
  int tensorSize = tensors_idx.size ();
  for (int i = 0; i < tensorSize; ++i) {
    data->interpreter->tensor (tensors_idx[i])->data.raw = nullptr;
  }

  if (status != kTfLiteOk) {
    g_critical ("Failed to invoke");
    return -1;
  }

  return 0;
}

/**
 * @brief Standard tensor_filter callback
 */
static int
edgetpu_getInputDim (const GstTensorFilterProperties *prop, void **private_data, GstTensorsInfo *info)
{
  pdata *data = (pdata *) *private_data;
  if (!data)
    return -1;
  gst_tensors_info_copy (info, &data->input);
  return 0;
}

/**
 * @brief Standard tensor_filter callback
 */
static int
edgetpu_getOutputDim (const GstTensorFilterProperties *prop, void **private_data, GstTensorsInfo *info)
{
  pdata *data = (pdata *) *private_data;
  if (!data)
    return -1;
  gst_tensors_info_copy (info, &data->output);
  return 0;
}

static gchar filter_subplugin_edgetpu[] = "edgetpu";

static GstTensorFilterFramework NNS_support_edgetpu = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = edgetpu_open,
  .close = edgetpu_close,
};

/**@brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_edgetpu (void)
{
  NNS_support_edgetpu.name = filter_subplugin_edgetpu;
  NNS_support_edgetpu.allow_in_place = FALSE;
  NNS_support_edgetpu.allocate_in_invoke = FALSE;
  NNS_support_edgetpu.run_without_model = FALSE;
  NNS_support_edgetpu.verify_model_path = FALSE;
  NNS_support_edgetpu.invoke_NN = edgetpu_invoke;
  NNS_support_edgetpu.getInputDimension = edgetpu_getInputDim;
  NNS_support_edgetpu.getOutputDimension = edgetpu_getOutputDim;

  nnstreamer_filter_probe (&NNS_support_edgetpu);
}

/** @brief Destruct the subplugin */
void
fini_filter_edgetpu (void)
{
  nnstreamer_filter_exit (NNS_support_edgetpu.name);
}
