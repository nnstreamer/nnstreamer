/**
 * NNStreamer API for Tensor_Filter Sub-Plugins
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file  nnstreamer_plugin_api_filter.h
 * @date  30 Jan 2019
 * @brief Mandatory APIs for NNStreamer Filter sub-plugins (No External Dependencies)
 * @see https://github.com/nnsuite/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNS_PLUGIN_API_FILTER_H__
#define __NNS_PLUGIN_API_FILTER_H__

#include "tensor_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief nnapi hw properties.
 */
typedef enum
{
  NNAPI_CPU = 0,
  NNAPI_GPU = 1,
  NNAPI_NPU = 2,

  NNAPI_UNKNOWN
} nnapi_hw;

/**
 * @brief GstTensorFilter's properties for NN framework (internal data structure)
 *
 * Because custom filters of tensor_filter may need to access internal data
 * of GstTensorFilter, we define this data structure here.
 */
typedef struct _GstTensorFilterProperties
{
  const char *fwname; /**< The name of NN Framework */
  int fw_opened; /**< TRUE IF open() is called or tried. Use int instead of gboolean because this is refered by custom plugins. */
  const char *model_file; /**< Filepath to the model file (as an argument for NNFW). char instead of gchar for non-glib custom plugins */
  const char *model_file_sub; /**< Filepath to the init model file (as an argument for NNFW). Some frameworks need this file to initialize the graph(caffe, caffe2) */

  int input_configured; /**< TRUE if input tensor is configured. Use int instead of gboolean because this is refered by custom plugins. */
  GstTensorsInfo input_meta; /**< configured input tensor info */

  int output_configured; /**< TRUE if output tensor is configured. Use int instead of gboolean because this is refered by custom plugins. */
  GstTensorsInfo output_meta; /**< configured output tensor info */

  const char *custom_properties; /**< sub-plugin specific custom property values in string */
  const char *nnapi;

} GstTensorFilterProperties;

/**
 * @brief Tensor_Filter Subplugin definition
 *
 * Common callback parameters:
 * prop Filter properties. Read Only.
 * private_data Subplugin's private data. Set this (*private_data = XXX) if you want to change filter->private_data.
 */
typedef struct _GstTensorFilterFramework
{
  char *name; /**< Name of the neural network framework, searchable by FRAMEWORK property */
  int allow_in_place; /**< TRUE(nonzero) if InPlace transfer of input-to-output is allowed. Not supported in main, yet */
  int allocate_in_invoke; /**< TRUE(nonzero) if invoke_NN is going to allocate outputptr by itself and return the address via outputptr. Do not change this value after cap negotiation is complete (or the stream has been started). */
  int run_without_model; /**< TRUE(nonzero) when the neural network framework does not need a model file. Tensor-filter will run invoke_NN without model. */

  int (*invoke_NN) (const GstTensorFilterProperties * prop, void **private_data,
      const GstTensorMemory * input, GstTensorMemory * output);
      /**< Mandatory callback. Invoke the given network model.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] input The array of input tensors. Allocated and filled by tensor_filter/main
       * @param[out] output The array of output tensors. Allocated by tensor_filter/main and to be filled by invoke_NN. If allocate_in_invoke is TRUE, sub-plugin should allocate the memory block for output tensor. (data in GstTensorMemory)
       * @return 0 if OK. non-zero if error.
       */

  int (*getInputDimension) (const GstTensorFilterProperties * prop,
      void **private_data, GstTensorsInfo * info);
      /**< Optional. Set NULL if not supported. Get dimension of input tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] info structure of tensor info (return value)
       * @return 0 if OK. non-zero if error.
       */

  int (*getOutputDimension) (const GstTensorFilterProperties * prop,
      void **private_data, GstTensorsInfo * info);
      /**< Optional. Set NULL if not supported. Get dimension of output tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] info structure of tensor info (return value)
       * @return 0 if OK. non-zero if error.
       */

  int (*setInputDimension) (const GstTensorFilterProperties * prop,
      void **private_data, const GstTensorsInfo * in_info,
      GstTensorsInfo * out_info);
      /**< Optional. Set Null if not supported. Tensor_Filter::main will
       * configure input dimension from pad-cap in run-time for the sub-plugin.
       * Then, the sub-plugin is required to return corresponding output dimension
       * If this is NULL, both getInput/OutputDimension must be non-NULL.
       *
       * When you use this, do NOT allocate or fix internal data structure based on it
       * until invoke is called. Gstreamer may try different dimensions before
       * settling down.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] in_info structure of input tensor info
       * @param[out] out_info structure of output tensor info (return value)
       * @return 0 if OK. non-zero if error.
       */

  int (*open) (const GstTensorFilterProperties * prop, void **private_data);
      /**< Optional. tensor_filter.c will call this before any of other callbacks and will call once before calling close
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, open() allocates memory for private_data.
       * @return 0 if ok. < 0 if error.
       */

  void (*close) (const GstTensorFilterProperties * prop, void **private_data);
      /**< Optional. tensor_filter.c will not call other callbacks after calling close. Free-ing private_data is this function's responsibility. Set NULL after that.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, close() frees private_data and set NULL.
       */

  void (*destroyNotify) (void * data);
      /**< Optional. tensor_filter.c will call it when 'allocate_in_invoke' flag of the framework is TRUE. Basically, it is called when the data element is destroyed. If it's set as NULL, g_free() will be used as a default. It will be helpful when the data pointer is included as an object of a nnfw. For instance, if the data pointer is removed when the object is gone, it occurs error. In this case, the objects should be maintained for a while first and destroyed when the data pointer is destroyed. Those kinds of logic could be defined at this method.
       *
       * @param[in] data the data element.
       */
} GstTensorFilterFramework;

/* extern functions for subplugin management, exist in tensor_filter.c */
/**
 * @brief Filter's sub-plugin should call this function to register itself.
 * @param[in] tfsp Tensor-Filter Sub-Plugin to be registered.
 * @return TRUE if registered. FALSE is failed or duplicated.
 */
extern int
nnstreamer_filter_probe (GstTensorFilterFramework * tfsp);

/**
 * @brief Filter's sub-plugin may call this to unregister itself.
 * @param[in] name The name of filter sub-plugin.
 */
extern void
nnstreamer_filter_exit (const char *name);

/**
 * @brief Find filter sub-plugin with the name.
 * @param[in] name The name of filter sub-plugin.
 * @return NULL if not found or the sub-plugin object has an error.
 */
extern const GstTensorFilterFramework *
nnstreamer_filter_find (const char *name);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_PLUGIN_API_FILTER_H__ */
