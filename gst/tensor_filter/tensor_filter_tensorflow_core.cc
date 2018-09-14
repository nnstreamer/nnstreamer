/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2018 HyoungJoo Ahn <hello.ahn@samsung.com>
 * Copyright (C) 2018 Jijoong Moon <jjioong.moon@samsung.com>
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
 * @file   tensor_filter_tensorflow_core.cc
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @date   08/02/2018
 * @brief  connection with tensorflow libraries.
 *
 * @bug     No known bugs.
 */

#include <sys/time.h>
#include <unistd.h>
#include <algorithm>

#include "tensor_filter_tensorflow_core.h"

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
 * @brief	TFCore creator
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFCore::TFCore (const char *_model_path)
{
  model_path = _model_path;

  loadModel ();
}

/**
 * @brief	TFCore Destructor
 * @return	Nothing
 */
TFCore::~TFCore ()
{
}

/**
 * @brief	get millisecond for time profiling.
 * @note	it returns the millisecond.
 * @param t	: the time struct.
 * @return the millisecond of t.
 */
double
TFCore::get_ms (struct timeval t)
{
  return (t.tv_sec * 1000000 + t.tv_usec);
}

/**
 * @brief	load the tf model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::loadModel ()
{

  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tensor_idx	: the index of the tensor
 * @param[out] type	: the data type of the input tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::getTensorType (int tensor_idx, tensor_type * type)
{

  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::getInputTensorDim (GstTensorsInfo * info)
{
  /**
   * @todo fill here
   */
  return 0;
}

/**
 * @brief	return the Dimension of Output Tensor.
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::getOutputTensorDim (GstTensorsInfo * info)
{
  /**
   * @todo fill here
   */
  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param tensor_idx	: the real index of model of the tensor
 * @param[out] dim	: the array of the tensor
 * @param[out] type	: the data type of the tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::getTensorDim (tensor_dim dim, tensor_type * type)
{

  return 0;
}

/**
 * @brief	return the number of Input Tensors.
 * @return	the number of Input Tensors.
 */
int
TFCore::getInputTensorSize ()
{
  return input_size;
}

/**
 * @brief	return the number of Output Tensors.
 * @return	the number of Output Tensors
 */
int
TFCore::getOutputTensorSize ()
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
TFCore::invoke (uint8_t * inptr, uint8_t ** outptr)
{
  return 0;
}

extern void *
tf_core_new (const char *_model_path)
{
  return new TFCore (_model_path);
}

/**
 * @brief	delete the TFCore class.
 * @param	tf	: the class object
 * @return	Nothing
 */
extern void
tf_core_delete (void *tf)
{
  TFCore *c = (TFCore *) tf;
  delete c;
}

/**
 * @brief	get model path
 * @param	tf	: the class object
 * @return	model path
 */
extern const char *
tf_core_getModelPath (void *tf)
{
  TFCore *c = (TFCore *) tf;
  return c->getModelPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of model
 * @param	tf	the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_getInputDim (void *tf, GstTensorsInfo * info)
{
  TFCore *c = (TFCore *) tf;
  return c->getInputTensorDim (info);
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tf	the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_getOutputDim (void *tf, GstTensorsInfo * info)
{
  TFCore *c = (TFCore *) tf;
  return c->getOutputTensorDim (info);
}

/**
 * @brief	get the size of Input Tensor of model
 * @param	tf	: the class object
 * @return	the number of Input Tensors.
 */
int
tf_core_getInputSize (void *tf)
{
  TFCore *c = (TFCore *) tf;
  return c->getInputTensorSize ();
}

/**
 * @brief	get the size of Output Tensor of model
 * @param	tf	: the class object
 * @return	the number of Output Tensors.
 */
int
tf_core_getOutputSize (void *tf)
{
  TFCore *c = (TFCore *) tf;
  return c->getOutputTensorSize ();
}

/**
 * @brief	invoke the model
 * @param	tf	: the class object
 * @param[in] inptr : The input tensor
 * @param[out]  outptr : The output tensor
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_invoke (void *tf, uint8_t * inptr, uint8_t ** outptr)
{
  TFCore *c = (TFCore *) tf;
  return c->invoke (inptr, outptr);
}
