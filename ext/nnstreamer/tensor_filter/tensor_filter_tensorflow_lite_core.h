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

#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
#include "tflite/ext/nnapi_delegate.h"
#endif

/**
 * @brief Wrapper class for TFLite Interpreter to support model switching
 */
class TFLiteInterpreter
{
public:
  TFLiteInterpreter ();
  ~TFLiteInterpreter ();

  int invoke (const GstTensorMemory * input, GstTensorMemory * output, bool use_nnapi);
  int loadModel (bool use_nnapi);
  void moveInternals (TFLiteInterpreter& interp);

  int setInputTensorProp ();
  int setOutputTensorProp ();
  int setInputTensorsInfo (const GstTensorsInfo * info);

  void setModelPath (const char *model_path);
  /** @brief get current model path */
  const char *getModelPath () { return model_path; }

  /** @brief return input tensor meta */
  const GstTensorsInfo* getInputTensorsInfo () { return &inputTensorMeta; }
  /** @brief return output tensor meta */
  const GstTensorsInfo* getOutputTensorsInfo () { return &outputTensorMeta; }

  /** @brief lock this interpreter */
  void lock () { g_mutex_lock (&mutex); }
  /** @brief unlock this interpreter */
  void unlock () { g_mutex_unlock (&mutex); }

private:
  GMutex mutex;
  char *model_path;

  std::unique_ptr <tflite::Interpreter> interpreter;
  std::unique_ptr <tflite::FlatBufferModel> model;
#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
  std::unique_ptr <nnfw::tflite::NNAPIDelegate> nnfw_delegate;
#endif

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */

  tensor_type getTensorType (TfLiteType tfType);
  int getTensorDim (int tensor_idx, tensor_dim dim);
  int setTensorProp (const std::vector<int> &tensor_idx_list,
      GstTensorsInfo * tensorMeta);
};

/**
 * @brief	ring cache structure
 */
class TFLiteCore
{
public:
  TFLiteCore (const char *_model_path, const char *accelerators);

  int init ();
  int loadModel ();
  gboolean compareModelPath (const char *model_path);
  int setInputTensorProp ();
  int setOutputTensorProp ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int setInputTensorDim (const GstTensorsInfo * info);
  int reloadModel (const char * model_path);
  int invoke (const GstTensorMemory * input, GstTensorMemory * output);

private:
  bool use_nnapi;
  accl_hw accelerator;

  TFLiteInterpreter interpreter;
  TFLiteInterpreter interpreter_sub;

  void setAccelerator (const char * accelerators);
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  void *tflite_core_new (const char *_model_path, const char *accelerators);
  void tflite_core_delete (void * tflite);
  int tflite_core_init (void * tflite);
  gboolean tflite_core_compareModelPath (void * tflite, const char * model_path);
  int tflite_core_getInputDim (void * tflite, GstTensorsInfo * info);
  int tflite_core_getOutputDim (void * tflite, GstTensorsInfo * info);
  int tflite_core_setInputDim (void * tflite, const GstTensorsInfo * in_info,
      GstTensorsInfo * out_info);
  int tflite_core_reloadModel (void * tflite, const char * model_path);
  int tflite_core_invoke (void * tflite, const GstTensorMemory * input,
      GstTensorMemory * output);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_FILTER_TENSORFLOW_LITE_CORE_H */
