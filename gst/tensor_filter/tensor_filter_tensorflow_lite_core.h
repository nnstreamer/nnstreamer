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
 * @bug     No known bugs.
 */
#ifndef TENSOR_FILTER_TENSORFLOW_LITE_H
#define TENSOR_FILTER_TENSORFLOW_LITE_H

#ifdef __cplusplus
#include <iostream>
#include <stdint.h>
#include <glib.h>

#include <tensorflow/contrib/lite/model.h>
#include <tensorflow/contrib/lite/optional_debug_tools.h>
#include <tensorflow/contrib/lite/string_util.h>
#include <tensorflow/contrib/lite/kernels/register.h>

#include <tensor_common.h>

/**
 * @brief	ring cache structure
 */
class TFLiteCore
{
public:
  TFLiteCore (const char *_model_path);
  ~TFLiteCore ();

  int init();
  int loadModel ();
  const char* getModelPath();
  int setInputTensorProp ();
  int setOutputTensorProp ();
  int getInputTensorSize ();
  int getOutputTensorSize ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int invoke (const GstTensorMemory * input, GstTensorMemory * output);

private:

  const char *model_path;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */

  std::unique_ptr < tflite::Interpreter > interpreter;
  std::unique_ptr < tflite::FlatBufferModel > model;

  double get_ms (struct timeval t);
  tensor_type getTensorType (TfLiteType tfType);
  int getTensorDim (int tensor_idx, tensor_dim dim);
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  extern void *tflite_core_new (const char *_model_path);
  extern void tflite_core_delete (void *tflite);
  extern int tflite_core_init (void *tflite);
  extern const char *tflite_core_getModelPath (void *tflite);
  extern int tflite_core_getInputDim (void *tflite, GstTensorsInfo * info);
  extern int tflite_core_getOutputDim (void *tflite, GstTensorsInfo * info);
  extern int tflite_core_getOutputSize (void *tflite);
  extern int tflite_core_getInputSize (void *tflite);
  extern int tflite_core_invoke (void *tflite, const GstTensorMemory * input,
      GstTensorMemory * output);

#ifdef __cplusplus
}
#endif

#endif
