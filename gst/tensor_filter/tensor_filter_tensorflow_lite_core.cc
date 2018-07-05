/**
 * Copyright (C) 2017 - 2018 Samsung Electronics Co., Ltd. All rights reserved.
 *
 * PROPRIETARY/CONFIDENTIAL
 *
 * This software is the confidential and proprietary information of
 * SAMSUNG ELECTRONICS ("Confidential Information"). You shall not
 * disclose such Confidential Information and shall use it only in
 * accordance with the terms of the license agreement you entered
 * into with SAMSUNG ELECTRONICS.  SAMSUNG make no representations
 * or warranties about the suitability of the software, either
 * express or implied, including but not limited to the implied
 * warranties of merchantability, fitness for a particular purpose,
 * or non-infringement. SAMSUNG shall not be liable for any damages
 * suffered by licensee as a result of using, modifying or distributing
 * this software or its derivatives.
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
