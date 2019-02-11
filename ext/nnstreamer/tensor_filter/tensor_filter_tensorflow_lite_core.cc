/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
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
 * @file   tensor_filter_tensorflow_lite_core.cc
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @date   7/5/2018
 * @brief  connection with tflite libraries.
 *
 * @bug     No known bugs.
 */

#include <unistd.h>
#include <algorithm>

#include <nnstreamer_plugin_api.h>
#include "tensor_filter_tensorflow_lite_core.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief	TFLiteCore creator
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFLiteCore::TFLiteCore (const char * _model_path)
{
  model_path = _model_path;

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
}

/**
 * @brief	TFLiteCore Destructor
 * @return	Nothing
 */
TFLiteCore::~TFLiteCore ()
{
  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
}

/**
 * @brief	initialize the object with tflite model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initialization of output tensor is failed.
 */
int
TFLiteCore::init ()
{
  if (loadModel ()) {
    GST_ERROR ("Failed to load model\n");
    return -1;
  }
  if (setInputTensorProp ()) {
    GST_ERROR ("Failed to initialize input tensor\n");
    return -2;
  }
  if (setOutputTensorProp ()) {
    GST_ERROR ("Failed to initialize output tensor\n");
    return -3;
  }
  return 0;
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char *
TFLiteCore::getModelPath ()
{
  return model_path;
}

/**
 * @brief	load the tflite model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::loadModel ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  if (!interpreter) {
    model =
        std::unique_ptr <tflite::FlatBufferModel>
        (tflite::FlatBufferModel::BuildFromFile (model_path));
    if (!model) {
      GST_ERROR ("Failed to mmap model\n");
      return -1;
    }
    /* If got any trouble at model, active below code. It'll be help to analyze. */
    /* model->error_reporter (); */

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder (*model, resolver) (&interpreter);
    if (!interpreter) {
      GST_ERROR ("Failed to construct interpreter\n");
      return -2;
    }

    /** set allocation type to dynamic for in/out tensors */
    int tensor_idx;

    int tensorSize = interpreter->inputs ().size ();
    for (int i = 0; i < tensorSize; ++i) {
      tensor_idx = interpreter->inputs ()[i];
      interpreter->tensor (tensor_idx)->allocation_type = kTfLiteDynamic;
    }

    tensorSize = interpreter->outputs ().size ();
    for (int i = 0; i < tensorSize; ++i) {
      tensor_idx = interpreter->outputs ()[i];
      interpreter->tensor (tensor_idx)->allocation_type = kTfLiteDynamic;
    }

    if (interpreter->AllocateTensors () != kTfLiteOk) {
      GST_ERROR ("Failed to allocate tensors\n");
      return -2;
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
 * @param tfType	: the defined type of Tensorflow Lite
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
TFLiteCore::getTensorType (TfLiteType tfType)
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
 * @brief extract and store the information of input tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setInputTensorProp ()
{
  auto input_idx_list = interpreter->inputs ();
  inputTensorMeta.num_tensors = input_idx_list.size ();

  for (int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    if (getTensorDim (input_idx_list[i], inputTensorMeta.info[i].dimension)) {
      GST_ERROR ("failed to get the dimension of input tensors");
      return -1;
    }
    inputTensorMeta.info[i].type =
        getTensorType (interpreter->tensor (input_idx_list[i])->type);

#if (DBG)
    gchar *dim_str =
        gst_tensor_get_dimension_string (inputTensorMeta.info[i].dimension);
    g_message ("inputTensorMeta[%d] >> type:%d, dim[%s]",
        i, inputTensorMeta.info[i].type, dim_str);
    g_free (dim_str);
#endif
  }
  return 0;
}

/**
 * @brief extract and store the information of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setOutputTensorProp ()
{
  auto output_idx_list = interpreter->outputs ();
  outputTensorMeta.num_tensors = output_idx_list.size ();

  for (int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    if (getTensorDim (output_idx_list[i], outputTensorMeta.info[i].dimension)) {
      GST_ERROR ("failed to get the dimension of output tensors");
      return -1;
    }
    outputTensorMeta.info[i].type =
        getTensorType (interpreter->tensor (output_idx_list[i])->type);

#if (DBG)
    gchar *dim_str =
        gst_tensor_get_dimension_string (outputTensorMeta.info[i].dimension);
    g_message ("outputTensorMeta[%d] >> type:%d, dim[%s]",
        i, outputTensorMeta.info[i].type, dim_str);
    g_free (dim_str);
#endif
  }
  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param tensor_idx	: the real index of model of the tensor
 * @param[out] dim	: the array of the tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getTensorDim (int tensor_idx, tensor_dim dim)
{
  int len = interpreter->tensor (tensor_idx)->dims->size;
  g_assert (len <= NNS_TENSOR_RANK_LIMIT);

  /* the order of dimension is reversed at CAPS negotiation */
  std::reverse_copy (interpreter->tensor (tensor_idx)->dims->data,
      interpreter->tensor (tensor_idx)->dims->data + len, dim);

  /* fill the remnants with 1 */
  for (int i = len; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 1;
  }

  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getInputTensorDim (GstTensorsInfo * info)
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
TFLiteCore::getOutputTensorDim (GstTensorsInfo * info)
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
TFLiteCore::invoke (const GstTensorMemory * input, GstTensorMemory * output)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  std::vector <int> tensors_idx;
  int tensor_idx;
  TfLiteTensor *tensor_ptr;

  for (int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->outputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == output[i].size);
    tensor_ptr->data.raw = (char *) output[i].data;
    tensors_idx.push_back (tensor_idx);
  }

  for (int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->inputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == input[i].size);
    tensor_ptr->data.raw = (char *) input[i].data;
    tensors_idx.push_back (tensor_idx);
  }

  if (interpreter->Invoke () != kTfLiteOk) {
    GST_ERROR ("Failed to invoke");
    return -3;
  }

  /** if it is not `nullptr`, tensorflow makes `free()` the memory itself. */
  int tensorSize = tensors_idx.size ();
  for (int i = 0; i < tensorSize; ++i) {
    interpreter->tensor (tensors_idx[i])->data.raw = nullptr;
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Invoke() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  return 0;
}

/**
 * @brief	call the creator of TFLiteCore class.
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @return	TFLiteCore class
 */
void *
tflite_core_new (const char * _model_path)
{
  return new TFLiteCore (_model_path);
}

/**
 * @brief	delete the TFLiteCore class.
 * @param	tflite	: the class object
 * @return	Nothing
 */
void
tflite_core_delete (void * tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  delete c;
}

/**
 * @brief	initialize the object with tflite model
 * @param	tflite	: the class object
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_init (void * tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->init ();
}

/**
 * @brief	get the model path
 * @param	tflite	: the class object
 * @return the model path.
 */
const char *
tflite_core_getModelPath (void * tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getModelPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of model
 * @param	tflite	: the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getInputDim (void * tflite, GstTensorsInfo * info)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getInputTensorDim (info);
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tflite	: the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getOutputDim (void * tflite, GstTensorsInfo * info)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getOutputTensorDim (info);
}

/**
 * @brief	invoke the model
 * @param	tflite	: the class object
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_invoke (void * tflite, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->invoke (input, output);
}
