/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
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
 * @file   tensor_filter_pytorch_core.h
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @date	 24 April 2019
 * @brief	 connection with pytorch libraries.
 * @bug    No known bugs except for NYI items
 */
#ifndef TENSOR_FILTER_PYTORCH_CORE_H
#define TENSOR_FILTER_PYTORCH_CORE_H


#include <glib.h>

#include "nnstreamer_plugin_api_filter.h"

#ifdef __cplusplus
#include <torch/script.h>

/**
 * @brief	ring cache structure
 */
class TorchCore
{
public:
  TorchCore (const char *_model_path);
  ~TorchCore ();

  int init (const GstTensorFilterProperties * prop,
      const gboolean torch_use_gpu);
  int loadModel ();
  const char *getModelPath ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int invoke (const GstTensorMemory * input, GstTensorMemory * output);

private:

  char *model_path;
  bool use_gpu;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */
  bool configured;
  bool first_run;           /**< must be reset after setting input info */

  std::shared_ptr < torch::jit::script::Module > model;

  tensor_type getTensorTypeFromTorch (torch::Dtype torchType);
  bool getTensorTypeToTorch (tensor_type tensorType, torch::Dtype * torchType);
  int validateOutputTensor (at::Tensor output);
  int fillTensorDim (torch::autograd::Variable tensor_meta, tensor_dim dim);
  int processIValue (torch::jit::IValue value, GstTensorMemory * output);
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif

  void *torch_core_new (const char *_model_path);
  void torch_core_delete (void *torch);
  int torch_core_init (void *torch, const GstTensorFilterProperties * prop,
      const gboolean torch_use_gpu);
  const char *torch_core_getModelPath (void *torch);
  int torch_core_getInputDim (void *torch, GstTensorsInfo * info);
  int torch_core_getOutputDim (void *torch, GstTensorsInfo * info);
  int torch_core_invoke (void *torch, const GstTensorMemory * input,
      GstTensorMemory * output);

#ifdef __cplusplus
}
#endif

#endif                          /* TENSOR_FILTER_PYTORCH_CORE_H */
