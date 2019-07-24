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
 * @file   tensor_filter_tensorflow_core.h
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @date   08/02/2018
 * @brief  connection with tensorflow libraries.
 *
 * @bug     No known bugs.
 */
#ifndef TENSOR_FILTER_TENSORFLOW_CORE_H
#define TENSOR_FILTER_TENSORFLOW_CORE_H

#include <glib.h>
#include <gst/gst.h>

#include "nnstreamer_plugin_api_filter.h"

#ifdef __cplusplus
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>

#include <tensorflow/c/c_api.h>

/**
 * @brief	Internal data structure for tensorflow
 */
typedef struct
{
  TF_DataType type;
  int rank;
  std::vector < std::int64_t > dims;
} tf_tensor_info_s;

/**
 * @brief	ring cache structure
 */
class TFCore
{
public:
  /**
   * member functions.
   */
  TFCore (const char * _model_path);
   ~TFCore ();

  int init (const GstTensorFilterProperties * prop);
  int loadModel ();
  const char *getModelPath ();

  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int run (const GstTensorMemory * input, GstTensorMemory * output);

  static std::map < void *, TF_Tensor * >outputTensorMap;

private:

  const char *model_path;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors from user input */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors from user input */

  std::vector < tf_tensor_info_s > input_tensor_info; /* hold information for TF */

  TF_Graph *graph;
  TF_Session *session;

  tensor_type getTensorTypeFromTF (TF_DataType tfType);
  TF_DataType getTensorTypeToTF (tensor_type tType);
  int validateTensor (const GstTensorsInfo * tensorInfo, int is_input);
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  void *tf_core_new (const char * _model_path);
  void tf_core_delete (void * tf);
  int tf_core_init (void * tf, const GstTensorFilterProperties * prop,
      const gboolean tf_mem_optmz);
  const char *tf_core_getModelPath (void * tf);
  int tf_core_getInputDim (void * tf, GstTensorsInfo * info);
  int tf_core_getOutputDim (void * tf, GstTensorsInfo * info);
  int tf_core_run (void * tf, const GstTensorMemory * input,
      GstTensorMemory * output);
  void tf_core_destroyNotify (void * data);

#ifdef __cplusplus
}
#endif

#endif                          /* TENSOR_FILTER_TENSORFLOW_CORE_H */
