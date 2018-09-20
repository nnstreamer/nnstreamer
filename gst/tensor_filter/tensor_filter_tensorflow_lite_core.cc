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
    /* If got any trouble at model, active below code. It'll be help to analyze. */
    /* model->error_reporter (); */

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

  for (int i = 0; i < inputTensorMeta.num_tensors; i++) {
    if (getTensorDim (input_idx_list[i], inputTensorMeta.info[i].dimension)) {
      return -1;
    }
    inputTensorMeta.info[i].type =
        getTensorType (interpreter->tensor (input_idx_list[i])->type);

#if (DBG)
    _print_log ("inputTensorMeta[%d] >> type:%d, dim[%d:%d:%d:%d], rank: %d",
        i, inputTensorMeta.info[i].type, inputTensorMeta.info[i].dimension[0],
        inputTensorMeta.info[i].dimension[1],
        inputTensorMeta.info[i].dimension[2],
        inputTensorMeta.info[i].dimension[3]);
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

  for (int i = 0; i < outputTensorMeta.num_tensors; i++) {
    if (getTensorDim (output_idx_list[i], outputTensorMeta.info[i].dimension)) {
      return -1;
    }
    outputTensorMeta.info[i].type =
        getTensorType (interpreter->tensor (output_idx_list[i])->type);

#if (DBG)
    _print_log ("outputTensorMeta[%d] >> type:%d, dim[%d:%d:%d:%d], rank: %d",
        i, outputTensorMeta.info[i].type, outputTensorMeta.info[i].dimension[0],
        outputTensorMeta.info[i].dimension[1],
        outputTensorMeta.info[i].dimension[2],
        outputTensorMeta.info[i].dimension[3]);
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
  return inputTensorMeta.num_tensors;
}

/**
 * @brief	return the number of Output Tensors.
 * @return	the number of Output Tensors
 */
int
TFLiteCore::getOutputTensorSize ()
{
  return outputTensorMeta.num_tensors;
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
  info->num_tensors = inputTensorMeta.num_tensors;
  memcpy (info->info, inputTensorMeta.info,
      sizeof (GstTensorInfo) * inputTensorMeta.num_tensors);
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
  info->num_tensors = outputTensorMeta.num_tensors;
  memcpy (info->info, outputTensorMeta.info,
      sizeof (GstTensorInfo) * outputTensorMeta.num_tensors);
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
  struct timeval start_time, stop_time;
  gettimeofday (&start_time, nullptr);
#endif

  int num_of_input[NNS_TENSOR_SIZE_LIMIT];
  for (int i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++) {
    num_of_input[i] = 1;
  }

  int sizeOfArray = NNS_TENSOR_RANK_LIMIT;

  if (interpreter->AllocateTensors () != kTfLiteOk) {
    printf ("Failed to allocate tensors\n");
    return -2;
  }

  for (int i = 0; i < getInputTensorSize (); i++) {
    int in_tensor = interpreter->inputs ()[i];

    for (int j = 0; j < sizeOfArray; j++) {
      num_of_input[i] *= inputTensorMeta.info[i].dimension[j];
    }

    for (int j = 0; j < num_of_input[i]; j++) {
      if (inputTensorMeta.info[i].type == _NNS_FLOAT32) {
        (interpreter->typed_tensor < float >(in_tensor))[j] =
            (((float *) input[i].data)[j] - 127.5f) / 127.5f;
      } else if (inputTensorMeta.info[i].type == _NNS_UINT8) {
        (interpreter->typed_tensor < uint8_t > (in_tensor))[j] =
            ((uint8_t *) input[i].data)[j];
      }
    }
  }

  if (interpreter->Invoke () != kTfLiteOk) {
    _print_log ("Failed to invoke");
    return -3;
  }

  for (int i = 0; i < outputTensorMeta.num_tensors; i++) {
    if (outputTensorMeta.info[i].type == _NNS_FLOAT32) {
      output[i].data = interpreter->typed_output_tensor < float >(i);
    } else if (outputTensorMeta.info[i].type == _NNS_UINT8) {
      output[i].data = interpreter->typed_output_tensor < uint8_t > (i);
    }
  }

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
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getInputDim (void *tflite, GstTensorsInfo * info)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  int ret = c->getInputTensorDim (info);
  return ret;
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tflite	: the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getOutputDim (void *tflite, GstTensorsInfo * info)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  int ret = c->getOutputTensorDim (info);
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
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_invoke (void *tflite, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->invoke (input, output);
}
