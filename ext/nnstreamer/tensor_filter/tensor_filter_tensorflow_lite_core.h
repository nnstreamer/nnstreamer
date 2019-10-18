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
#ifndef TENSOR_FILTER_TENSORFLOW_LITE_CORE_H
#define TENSOR_FILTER_TENSORFLOW_LITE_CORE_H

#include <glib.h>

#include "nnstreamer_plugin_api_filter.h"

#ifdef __cplusplus
#include <iostream>

#include <tensorflow/contrib/lite/model.h>
#include <tensorflow/contrib/lite/kernels/register.h>

#ifdef ENABLE_NNFW
#include "tflite/ext/nnapi_delegate.h"
#endif

/**
 * @brief	ring cache structure
 */
class TFLiteCore
{
public:
  TFLiteCore (const char *_model_path, nnapi_hw hw);
  ~TFLiteCore ();

  int init ();
  int loadModel ();
  const char* getModelPath ();
  int setInputTensorProp ();
  int setOutputTensorProp ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int setInputTensorDim (const GstTensorsInfo * info);
  int invoke (const GstTensorMemory * input, GstTensorMemory * output);

private:

  const char *model_path;
  bool use_nnapi;
  nnapi_hw accel;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */

  std::unique_ptr <tflite::Interpreter> interpreter;
  std::unique_ptr <tflite::FlatBufferModel> model;

#ifdef ENABLE_NNFW
  std::unique_ptr <nnfw::tflite::NNAPIDelegate> nnfw_delegate;
#endif

  tensor_type getTensorType (TfLiteType tfType);
  int getTensorDim (int tensor_idx, tensor_dim dim);

  int setTensorProp (const std::vector<int> &tensor_idx_list,
      GstTensorsInfo * tensorMeta);
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  void *tflite_core_new (const char *_model_path, nnapi_hw hw);
  void tflite_core_delete (void * tflite);
  int tflite_core_init (void * tflite);
  const char *tflite_core_getModelPath (void * tflite);
  int tflite_core_getInputDim (void * tflite, GstTensorsInfo * info);
  int tflite_core_getOutputDim (void * tflite, GstTensorsInfo * info);
  int tflite_core_setInputDim (void * tflite, const GstTensorsInfo * in_info,
      GstTensorsInfo * out_info);
  int tflite_core_invoke (void * tflite, const GstTensorMemory * input,
      GstTensorMemory * output);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_FILTER_TENSORFLOW_LITE_CORE_H */
