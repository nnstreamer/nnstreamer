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
  loadModel ();
  setInputTensorProp ();
  setOutputTensorProp ();

}

/**
 * @brief	TFLiteCore Destructor
 * @return	Nothing
 */
TFLiteCore::~TFLiteCore ()
{
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
#if (DBG)
  gettimeofday (&stop_time, nullptr);
  _print_log ("Model is Loaded: %lf",
      (get_ms (stop_time) - get_ms (start_time)) / 1000);
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tfType	: the defined type of Tensorflow Lite
 * @return the enum of defined _NNS_TYPE
 */
_nns_tensor_type TFLiteCore::getTensorType (TfLiteType tfType)
{
  switch (tfType) {
    case kTfLiteFloat32:
      return _NNS_FLOAT32;
      break;
    case kTfLiteUInt8:
      return _NNS_UINT8;
      break;
    case kTfLiteInt32:
      return _NNS_INT32;
      break;
    case kTfLiteBool:
      return _NNS_INT8;
      break;
    case kTfLiteInt64:
    case kTfLiteString:
    default:
      return _NNS_END;
      break;
  }
}

/**
 * @brief extract and store the information of input tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setInputTensorProp ()
{
  auto input_idx_list = interpreter->inputs ();
  inputTensorSize = input_idx_list.size ();

  for (int i = 0; i < inputTensorSize; i++) {
    if (getTensorDim (input_idx_list[i], inputDimension[i])) {
      return -1;
    }
    inputType[i] =
        getTensorType (interpreter->tensor (input_idx_list[i])->type);
  }
#if (DBG)
  if (ret) {
    _print_log ("Failed to getInputTensorDim");
  } else {
    _print_log ("InputTensorDim idx[%d] type[%d] dim[%d:%d:%d:%d]",
        idx, *type, dim[0], dim[1], dim[2], dim[3]);
  }
#endif
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
  outputTensorSize = output_idx_list.size ();

  for (int i = 0; i < outputTensorSize; i++) {
    if (getTensorDim (output_idx_list[i], outputDimension[i])) {
      return -1;
    }
    outputType[i] =
        getTensorType (interpreter->tensor (output_idx_list[i])->type);
  }
#if (DBG)
  if (ret) {
    _print_log ("Failed to getOutputTensorDim");
  } else {
    _print_log ("OutputTensorDim idx[%d] type[%d] dim[%d:%d:%d:%d]",
        idx, *type, dim[0], dim[1], dim[2], dim[3]);
  }
#endif
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
  return inputTensorSize;
}

/**
 * @brief	return the number of Output Tensors.
 * @return	the number of Output Tensors
 */
int
TFLiteCore::getOutputTensorSize ()
{
  return outputTensorSize;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] dim	: the array of the input tensors
 * @param[out] type	: the data type of the input tensors
 * @todo : return whole array rather than index 0
 * @return the number of input tensors;
 */
int
TFLiteCore::getInputTensorDim (tensor_dim dim, tensor_type * type)
{
  memcpy (dim, inputDimension[0], sizeof (tensor_dim));
  *type = inputType[0];
  return inputTensorSize;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] dim	: the array of the tensors
 * @param[out] type	: the data type of the tensors
 * @todo : return whole array rather than index 0
 * @return the number of output tensors;
 */
int
TFLiteCore::getOutputTensorDim (tensor_dim dim, tensor_type * type)
{
  memcpy (dim, outputDimension[0], sizeof (tensor_dim));
  *type = outputType[0];
  return outputTensorSize;
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

  for (int i = 0; i < sizeOfArray; i++) {
    output_number_of_pixels *= inputDimension[0][i];
  }

  for (int i = 0; i < getInputTensorSize (); i++) {
    int input = interpreter->inputs ()[i];

    if (interpreter->AllocateTensors () != kTfLiteOk) {
      _print_log ("Failed to allocate tensors");
      return -2;
    }

    inputTensors[0] = inptr;
    for (int j = 0; j < output_number_of_pixels; j++) {
      if (inputType[i] == _NNS_FLOAT32) {
        (interpreter->typed_tensor < float >(input))[j] =
            ((float) inputTensors[i][j] - 127.5f) / 127.5f;
      } else if (inputType[i] == _NNS_UINT8) {
        (interpreter->typed_tensor < uint8_t > (input))[j] = inputTensors[i][j];
      }
    }
  }

  if (interpreter->Invoke () != kTfLiteOk) {
    _print_log ("Failed to invoke");
    return -3;
  }

  for (int i = 0; i < outputTensorSize; i++) {

    if (outputType[i] == _NNS_FLOAT32) {
      outputTensors[i] =
          (uint8_t *) interpreter->typed_output_tensor < float >(i);
    } else if (outputType[i] == _NNS_UINT8) {
      outputTensors[i] = interpreter->typed_output_tensor < uint8_t > (i);
    }
  }
  *outptr = outputTensors[0];

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
 * @brief	get the Dimension of Input Tensor of model
 * @param	tflite	: the class object
 * @param[out] dim	: the array of the input tensor
 * @param[out] type	: the data type of the input tensor
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getInputDim (void *tflite, tensor_dim dim, tensor_type * type)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  int ret = c->getInputTensorDim (dim, type);
  return ret;
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tflite	: the class object
 * @param[out] dim	: the array of the output tensor
 * @param[out] type	: the data type of the output tensor
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getOutputDim (void *tflite, tensor_dim dim, tensor_type * type)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  int ret = c->getOutputTensorDim (dim, type);
  return ret;
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
