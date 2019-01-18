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
#ifndef TENSOR_FILTER_TENSORFLOW_H
#define TENSOR_FILTER_TENSORFLOW_H

#ifdef __cplusplus
#include <glib.h>
#include <gst/gst.h>
#include <setjmp.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/util/command_line_flags.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/tools/graph_transforms/transform_utils.h>

#include <tensor_typedef.h>

using namespace tensorflow;

/**
 * @brief	ring cache structure
 */
class TFCore
{
public:
  /**
   * member functions.
   */
  TFCore (const char *_model_path);
   ~TFCore ();

  int init(const GstTensorFilterProperties * prop);
  int loadModel ();
  const char* getModelPath();
  int setInputTensorProp ();
  int setOutputTensorProp ();

  int getInputTensorSize ();
  int getOutputTensorSize ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int run (const GstTensorMemory * input, GstTensorMemory * output);

  static std::map<void*, Tensor> outputTensorMap;

private:

  const char *model_path;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */

  int inputTensorRank[NNS_TENSOR_SIZE_LIMIT];
  int outputTensorRank[NNS_TENSOR_SIZE_LIMIT];

  Session * session;

  tensor_type getTensorTypeFromTF (DataType tfType);
  DataType getTensorTypeToTF (tensor_type tType);
  int setTensorProp (GstTensorsInfo * dest, const GstTensorsInfo * src);
  int inputTensorValidation (const std::vector<const NodeDef*> &placeholders);
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  extern void *tf_core_new (const char *_model_path);
  extern void tf_core_delete (void *tf);
  extern int tf_core_init (void *tf, const GstTensorFilterProperties * prop);
  extern const char *tf_core_getModelPath (void *tf);
  extern int tf_core_getInputDim (void *tf, GstTensorsInfo * info);
  extern int tf_core_getOutputDim (void *tf, GstTensorsInfo * info);
  extern int tf_core_run (void *tf, const GstTensorMemory * input,
      GstTensorMemory * output);
  extern void tf_core_destroyNotify (void * data);

#ifdef __cplusplus
}
#endif

#endif
