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
 * @file   tensor_filter_tensorflow_lite_core.h
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @date   7/5/2018
 * @brief  connection with tflite libraries.
 *
 * @bug     No know bugs.
 * @todo    Invoke() should be implemented.
 * @todo    If it is required, class will be implemented as a singleton.
 */
#ifndef TENSOR_FILTER_TENSORFLOW_LITE_H
#define TENSOR_FILTER_TENSORFLOW_LITE_H

#ifdef __cplusplus
#include <iostream>

#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/kernels/register.h"

/**
 * @brief ring cache structure
 */
class TFLiteCore
{
public:
  /**
   * member functions.
   */
  TFLiteCore (char *_model_path);
  char *getModelPath ()
  {
    return model_path;
  }
  int loadModel ();
  const char *getInputTensorName ();
  const char *getOutputTensorName ();

  /**
   * @brief @todo fill this in
   */
  int getInputTensorSize ()
  {
    return input_size;
  }

  /**
   * @brief @todo fill this in
   */
  int getOutputTensorSize ()
  {
    return output_size;
  }
  int *getInputTensorDim ();
  int *getOutputTensorDim ();

private:
  /**
   * member variables.
   */
  char *model_path;
  int tensor_size;
  int node_size;
  int input_size;
  int output_size;
  const char *input_name;
  const char *output_name;
  int input_idx;
  int output_idx;
  std::unique_ptr < tflite::Interpreter > interpreter;
  std::unique_ptr < tflite::FlatBufferModel > model;
};

/**
 * @brief	TFLiteCore creator
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFLiteCore::TFLiteCore (char *_model_path)
{
  model_path = _model_path;
  loadModel ();
}

/**
 * @brief	load the tflite model
 * @note	the model will be loaded
 * @return	Nothing
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
  input_name = interpreter->GetInputName (0);
  output_size = interpreter->outputs ().size ();
  output_name = interpreter->GetOutputName (0);

  int t_size = interpreter->tensors_size ();
  for (int i = 0; i < t_size; i++) {
    if (strcmp (interpreter->tensor (i)->name,
            interpreter->GetInputName (0)) == 0)
      input_idx = i;
    if (strcmp (interpreter->tensor (i)->name,
            interpreter->GetOutputName (0)) == 0)
      output_idx = i;
  }
  return 1;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @return the array of integer.
 */
int *
TFLiteCore::getInputTensorDim ()
{
  return interpreter->tensor (input_idx)->dims->data;
}

/**
 * @brief	return the Dimension of Output Tensor.
 * @return the array of integer.
 */
int *
TFLiteCore::getOutputTensorDim ()
{
  return interpreter->tensor (output_idx)->dims->data;
}

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  extern void *tflite_core_new (char *_model_path);
  extern void tflite_core_delete (void *tflite);
  extern char *tflite_core_getModelPath (void *tflite);
  extern int *tflite_core_getInputDim (void *tflite);
  extern int *tflite_core_getOutputDim (void *tflite);
  extern int tflite_core_getInputSize (void *tflite);
  extern int tflite_core_getOutputSize (void *tflite);

#ifdef __cplusplus
}
#endif

#endif
