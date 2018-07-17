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
 * @todo    If it is required, class will be implemented as a singleton.
 */

#include "tensor_filter_tensorflow_lite_core.h"

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
 * @brief	load the tflite model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::loadModel ()
{
  if (!interpreter) {
    model =
        std::unique_ptr < tflite::FlatBufferModel >
        (tflite::FlatBufferModel::BuildFromFile (model_path));
    if (!model) {
      std::cout << "Failed to mmap model" << std::endl;
      return -1;
    }
    model->error_reporter ();
    std::cout << "model loaded" << std::endl;

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder (*model, resolver) (&interpreter);
    if (!interpreter) {
      std::cout << "Failed to construct interpreter" << std::endl;
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
  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param idx	: the index of the input tensor
 * @param[out] dim	: the array of the input tensor
 * @param[out] len	: the length of the input tensor array
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getInputTensorDim (int idx, int **dim, int *len)
{
  if (idx >= input_size) {
    return -1;
  }
  *dim = interpreter->tensor (input_idx_list[idx])->dims->data;
  *len = interpreter->tensor (input_idx_list[idx])->dims->size;

  return 0;
}

/**
 * @brief	return the Dimension of Output Tensor.
 * @param idx	: the index of the output tensor
 * @param[out] dim	: the array of the output tensor
 * @param[out] len	: the length of the output tensor array
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getOutputTensorDim (int idx, int **dim, int *len)
{
  if (idx >= output_size) {
    return -1;
  }
  *dim = interpreter->tensor (output_idx_list[idx])->dims->data;
  *len = interpreter->tensor (output_idx_list[idx])->dims->size;

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
  int output_number_of_pixels = 1;

  int sizeOfArray = 0;
  int *inputTensorDim;
  int ret = getInputTensorDim (0, &inputTensorDim, &sizeOfArray);
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
 * @param[out] len	: the length of the input tensor array
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getInputDim (void *tflite, int idx, int **dim, int *len)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getInputTensorDim (idx, dim, len);
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tflite	: the class object
 * @param idx	: the index of the output tensor
 * @param[out] dim	: the array of the output tensor
 * @param[out] len	: the length of the output tensor array
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getOutputDim (void *tflite, int idx, int **dim, int *len)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getOutputTensorDim (idx, dim, len);
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
