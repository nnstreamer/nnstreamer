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
 * @brief	 connection with tflite libraries.
 *
 * @bug     No know bugs.
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
 * @brief	ring cache structure
 */
class TFLiteCore
{
public:
  /**
   * member functions.
   */
  TFLiteCore (const char *_model_path);
   ~TFLiteCore ();

  /**
   * @brief	get the model path.
   * @return	saved model path.
   */
  const char *getModelPath ()
  {
    return model_path;
  }
  int loadModel ();
  const char *getInputTensorName ();
  const char *getOutputTensorName ();

  int getInputTensorSize ();
  int getOutputTensorSize ();
  int getInputTensorDim (int idx, int **dim, int *len);
  int getOutputTensorDim (int idx, int **dim, int *len);
  int getInputTensorDimSize ();
  int getOutputTensorDimSize ();
  int invoke (uint8_t * inptr, uint8_t ** outptr);

private:
  /**
   * member variables.
   */
  const char *model_path;
  int tensor_size;
  int node_size;
  int input_size;
  int output_size;
  int *input_idx_list;
  int *output_idx_list;
  int input_idx_list_len;
  int output_idx_list_len;
  std::unique_ptr < tflite::Interpreter > interpreter;
  std::unique_ptr < tflite::FlatBufferModel > model;
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  extern void *tflite_core_new (const char *_model_path);
  extern void tflite_core_delete (void *tflite);
  extern const char *tflite_core_getModelPath (void *tflite);
  extern int tflite_core_getInputDim (void *tflite, int idx, int **dim,
      int *len);
  extern int tflite_core_getOutputDim (void *tflite, int idx, int **dim,
      int *len);
  extern int tflite_core_getInputSize (void *tflite);
  extern int tflite_core_getOutputSize (void *tflite);
  extern int tflite_core_invoke (void *tflite, uint8_t * inptr,
      uint8_t ** outptr);

#ifdef __cplusplus
}
#endif

#endif
