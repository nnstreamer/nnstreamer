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

#include <sys/time.h>
#include <unistd.h>
#include <algorithm>

#include "tensor_filter_tensorflow_lite_core.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Macro for debug message.
 */
#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

/**
 * @brief	TFLiteCore creator
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFLiteCore::TFLiteCore (const char *_model_path)
{
  model_path = _model_path;
  input_idx_list_len = 0;
  output_idx_list_len = 0;

  loadModel ();
}

/**
 * @brief	TFLiteCore Destructor
 * @return	Nothing
 */
TFLiteCore::~TFLiteCore ()
{
  delete[]input_idx_list;
  delete[]output_idx_list;
}

/**
 * @brief	get millisecond for time profiling.
 * @note	it returns the millisecond.
 * @param t	: the time struct.
 * @return the millisecond of t.
 */
double
TFLiteCore::get_ms (struct timeval t)
{
  return (t.tv_sec * 1000000 + t.tv_usec);
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
  struct timeval start_time, stop_time;
  gettimeofday (&start_time, nullptr);
#endif

  if (!interpreter) {
    model =
        std::unique_ptr < tflite::FlatBufferModel >
        (tflite::FlatBufferModel::BuildFromFile (model_path));
    if (!model) {
      _print_log ("Failed to mmap model\n");
      return -1;
    }
    model->error_reporter ();

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder (*model, resolver) (&interpreter);
    if (!interpreter) {
      _print_log ("Failed to construct interpreter\n");
      return -2;
    }
  }
  // fill class parameters
  tensor_size = interpreter->tensors_size ();
  node_size = interpreter->nodes_size ();
  input_size = interpreter->inputs ().size ();
  output_size = interpreter->outputs ().size ();

  // allocate the idx of input/output tensors
  // it could be used for get name of the tensors by using 'interpreter->GetOutputName(0);'
  input_idx_list = new int[input_size];
  output_idx_list = new int[output_size];

  int t_size = interpreter->tensors_size ();
  for (int i = 0; i < t_size; i++) {
    for (int j = 0; j < input_size; j++) {
      if (strcmp (interpreter->tensor (i)->name,
              interpreter->GetInputName (j)) == 0)
        input_idx_list[input_idx_list_len++] = i;
    }
    for (int j = 0; j < output_size; j++) {
      if (strcmp (interpreter->tensor (i)->name,
              interpreter->GetOutputName (j)) == 0)
        output_idx_list[output_idx_list_len++] = i;
    }
  }

#if (DBG)
  gettimeofday (&stop_time, nullptr);
  _print_log ("Model is Loaded: %lf",
      (get_ms (stop_time) - get_ms (start_time)) / 1000);
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tensor_idx	: the index of the tensor
 * @param[out] type	: the data type of the input tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getTensorType (int tensor_idx, tensor_type * type)
{
  switch (interpreter->tensor (tensor_idx)->type) {
    case kTfLiteFloat32:
      *type = _NNS_FLOAT32;
      break;
    case kTfLiteUInt8:
      *type = _NNS_UINT8;
      break;
    case kTfLiteInt32:
      *type = _NNS_INT32;
      break;
    case kTfLiteBool:
      *type = _NNS_INT8;
      break;
    case kTfLiteInt64:
    case kTfLiteString:
    default:
      *type = _NNS_END;
      return -1;
      break;
  }
  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param idx	: the index of the input tensor
 * @param[out] dim	: the array of the input tensor
 * @param[out] type	: the data type of the input tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getInputTensorDim (int idx, tensor_dim dim, tensor_type * type)
{
  if (idx >= input_size) {
    return -1;
  }
  int ret = getTensorDim (input_idx_list[idx], dim, type);
  return ret;
}

/**
 * @brief	return the Dimension of Output Tensor.
 * @param idx	: the index of the output tensor
 * @param[out] dim	: the array of the output tensor
 * @param[out] type	: the data type of the output tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getOutputTensorDim (int idx, tensor_dim dim, tensor_type * type)
{
  if (idx >= output_size) {
    return -1;
  }
  int ret = getTensorDim (output_idx_list[idx], dim, type);
  return ret;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param tensor_idx	: the real index of model of the tensor
 * @param[out] dim	: the array of the tensor
 * @param[out] type	: the data type of the tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getTensorDim (int tensor_idx, tensor_dim dim, tensor_type * type)
{

  if (getTensorType (tensor_idx, type)) {
    return -2;
  }

  int len = interpreter->tensor (tensor_idx)->dims->size;
  g_assert (len <= NNS_TENSOR_RANK_LIMIT);

  /* the order of dimension is reversed at CAPS negotiation */
  std::reverse_copy (interpreter->tensor (tensor_idx)->dims->data,
      interpreter->tensor (tensor_idx)->dims->data + len, dim);

  /* fill the remnants with 1 */
  for (int i = len; i < NNS_TENSOR_RANK_LIMIT; i++) {
    dim[i] = 1;
  }

  return 0;
}

/**
 * @brief	return the number of Input Tensors.
 * @return	the number of Input Tensors.
 */
int
TFLiteCore::getInputTensorSize ()
{
  return input_size;
}

/**
 * @brief	return the number of Output Tensors.
 * @return	the number of Output Tensors
 */
int
TFLiteCore::getOutputTensorSize ()
{
  return output_size;
}

/**
 * @brief	run the model with the input.
 * @param[in] inptr : The input tensor
 * @param[out]  outptr : The output tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::invoke (uint8_t * inptr, uint8_t ** outptr)
{
#if (DBG)
  struct timeval start_time, stop_time;
  gettimeofday (&start_time, nullptr);
#endif

  int output_number_of_pixels = 1;

  int sizeOfArray = NNS_TENSOR_RANK_LIMIT;
  tensor_type type;
  tensor_dim inputTensorDim;
  int ret = getInputTensorDim (0, inputTensorDim, &type);
  if (ret) {
    return -1;
  }
  for (int i = 0; i < sizeOfArray; i++) {
    output_number_of_pixels *= inputTensorDim[i];
  }

  int input = interpreter->inputs ()[0];

  if (interpreter->AllocateTensors () != kTfLiteOk) {
    std::cout << "Failed to allocate tensors!" << std::endl;
    return -2;
  }

  for (int i = 0; i < output_number_of_pixels; i++) {
    (interpreter->typed_tensor < uint8_t > (input))[i] = (uint8_t) inptr[i];
  }

  if (interpreter->Invoke () != kTfLiteOk) {
    return -3;
  }

  *outptr = interpreter->typed_output_tensor < uint8_t > (0);

#if (DBG)
  gettimeofday (&stop_time, nullptr);
  _print_log ("Invoke() is finished: %lf",
      (get_ms (stop_time) - get_ms (start_time)) / 1000);
#endif

  return 0;
}

/**
 * @brief	call the creator of TFLiteCore class.
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @return	TFLiteCore class
 */
extern void *
tflite_core_new (const char *_model_path)
{
  return new TFLiteCore (_model_path);
}

/**
 * @brief	delete the TFLiteCore class.
 * @param	tflite	: the class object
 * @return	Nothing
 */
extern void
tflite_core_delete (void *tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  delete c;
}

/**
 * @brief	get model path
 * @param	tflite	: the class object
 * @return	model path
 */
extern const char *
tflite_core_getModelPath (void *tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getModelPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of model
 * @param	tflite	: the class object
 * @param idx	: the index of the input tensor
 * @param[out] dim	: the array of the input tensor
 * @param[out] type	: the data type of the input tensor
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getInputDim (void *tflite, int idx, tensor_dim dim,
    tensor_type * type)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getInputTensorDim (idx, dim, type);
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tflite	: the class object
 * @param idx	: the index of the output tensor
 * @param[out] dim	: the array of the output tensor
 * @param[out] type	: the data type of the output tensor
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getOutputDim (void *tflite, int idx, tensor_dim dim,
    tensor_type * type)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getOutputTensorDim (idx, dim, type);
}

/**
 * @brief	get the size of Input Tensor of model
 * @param	tflite	: the class object
 * @return	the number of Input Tensors.
 */
int
tflite_core_getInputSize (void *tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getInputTensorSize ();
}

/**
 * @brief	get the size of Output Tensor of model
 * @param	tflite	: the class object
 * @return	the number of Output Tensors.
 */
int
tflite_core_getOutputSize (void *tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getOutputTensorSize ();
}

/**
 * @brief	invoke the model
 * @param	tflite	: the class object
 * @param[in] inptr : The input tensor
 * @param[out]  outptr : The output tensor
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_invoke (void *tflite, uint8_t * inptr, uint8_t ** outptr)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->invoke (inptr, outptr);
}
