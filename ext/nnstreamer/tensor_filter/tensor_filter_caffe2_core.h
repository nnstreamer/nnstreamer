/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2019 HyoungJoo Ahn <hello.ahn@samsung.com>
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
 * @file   tensor_filter_caffe2_core.h
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @date   05/31/2019
 * @brief  connection with caffe2 libraries.
 *
 * @bug     No known bugs.
 */
#ifndef TENSOR_FILTER_CAFFE2_CORE_H
#define TENSOR_FILTER_CAFFE2_CORE_H

#include "nnstreamer_plugin_api_filter.h"

#ifdef __cplusplus
#include <iostream>

#include "caffe2/core/workspace.h"
#include "caffe2/core/init.h"

using namespace caffe2;

/**
 * @brief	ring cache structure
 */
class Caffe2Core
{
public:
  Caffe2Core (const char * _model_path, const char * _model_path_sub);
  ~Caffe2Core ();

  int init (const GstTensorFilterProperties * prop);
  int loadModels ();
  const char* getPredModelPath ();
  const char* getInitModelPath ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int run (const GstTensorMemory * input, GstTensorMemory * output);

private:

  const char *init_model_path;
  const char *pred_model_path;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */

  Workspace workSpace;
  NetDef initNet, predictNet;
  static std::map <char*, Tensor*> inputTensorMap;

  int initInputTensor ();
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  void *caffe2_core_new (const char *_model_path, const char *_model_path_sub);
  void caffe2_core_delete (void * caffe2);
  int caffe2_core_init (void * caffe2, const GstTensorFilterProperties * prop);
  const char *caffe2_core_getInitModelPath (void * caffe2);
  const char *caffe2_core_getPredModelPath (void * caffe2);
  int caffe2_core_getInputDim (void * caffe2, GstTensorsInfo * info);
  int caffe2_core_getOutputDim (void * caffe2, GstTensorsInfo * info);
  int caffe2_core_run (void * caffe2, const GstTensorMemory * input,
      GstTensorMemory * output);
  void caffe2_core_destroyNotify (void * data);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_FILTER_CAFFE2_CORE_H */
