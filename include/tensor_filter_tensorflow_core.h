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
 * @file   tensor_filter_tensorflow_lite_core.h
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
#include <iostream>
#include <stdint.h>
#include <glib.h>

#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/tensor_shape.h>

#include "tensor_typedef.h"

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

  double get_ms (struct timeval t);
  int getInputTensorSize ();
  int getOutputTensorSize ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int getInputTensorDimSize ();
  int getOutputTensorDimSize ();
  int invoke (const GstTensorMemory * input, GstTensorMemory * output);

private:
  /**
   * member variables.
   */
  const char *model_path;
  int tensor_size;
  int node_size;
  int input_size;
  int output_size;
  int getTensorType (int tensor_idx, tensor_type * type);
  int getTensorDim (tensor_dim dim, tensor_type * type);
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  extern void *tf_core_new (const char *_model_path);
  extern void tf_core_delete (void *tf);
  extern const char *tf_core_getModelPath (void *tf);
  extern int tf_core_getInputDim (void *tf, GstTensorsInfo * info);
  extern int tf_core_getOutputDim (void *tf, GstTensorsInfo * info);
  extern int tf_core_getInputSize (void *tf);
  extern int tf_core_getOutputSize (void *tf);
  extern int tf_core_invoke (void *tf, const GstTensorMemory * input, GstTensorMemory * output);

#ifdef __cplusplus
}
#endif

#endif
