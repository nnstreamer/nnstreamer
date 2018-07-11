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
 * @bug     No know bugs.
 * @todo    Invoke() should be implemented.
 * @todo    If it is required, class will be implemented as a singleton.
 */
#include "tensor_filter_tensorflow_lite_core.h"

/**
 * @brief	call the creator of TFLiteCore class.
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @return	TFLiteCore class
 */
extern void *
tflite_core_new (char *_model_path)
{
  return new TFLiteCore (_model_path);
}

/**
 * @brief	delete the TFLiteCore class.
 * @param	_tflite	: the class object
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
 * @param	_tflite	: the class object
 * @return	model path
 */
extern char *
tflite_core_getModelPath (void *tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getModelPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of model
 * @param	_tflite	: the class object
 * @return	the input dimension
 */
int *
tflite_core_getInputDim (void *tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getInputTensorDim ();
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	_tflite	: the class object
 * @return	the output dimension
 */
int *
tflite_core_getOutputDim (void *tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getOutputTensorDim ();
}

/**
 * @brief	get the size of Input Tensor of model
 * @param	_tflite	: the class object
 * @return	how many input tensors are
 */
int
tflite_core_getInputSize (void *tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getInputTensorSize ();
}

/**
 * @brief	get the size of Output Tensor of model
 * @param	_tflite	: the class object
 * @return	how many output tensors are
 */
int
tflite_core_getOutputSize (void *tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getOutputTensorSize ();
}
