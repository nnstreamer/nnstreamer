/**
 * GStreamer Tensor_Filter, Tensorflow Module
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
 * @file	tensor_filter_tensorflow.c
 * @date	02 Aug 2018
 * @brief	Tensorflow module for tensor_filter gstreamer plugin
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include "tensor_filter.h"
#include "tensor_filter_tensorflow_core.h"
#include <glib.h>

/**
 * @brief internal data of tensorflow lite
 */
struct _Tf_data
{
  void *tf_private_data;
};
typedef struct _Tf_data tf_data;

/**
 * @brief Load tensorflow lite modelfile
 * @param filter : tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded). -1 if error
 */
static int
tf_loadModelFile (const GstTensorFilter * filter, void **private_data)
{
  tf_data *tf;
  if (filter->privateData != NULL) {
    /** @todo : Check the integrity of filter->data and filter->model_file, nnfw */
    return 1;
  }
  tf = g_new0 (tf_data, 1); /** initialize tf Fill Zero! */
  *private_data = tf;
  tf->tf_private_data = tf_core_new (filter->prop.model_file);
  if (tf->tf_private_data) {
    return 0;
  } else {
    return -1;
  }
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param filter : tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 */
static void
tf_open (const GstTensorFilter * filter, void **private_data)
{
  int retval = tf_loadModelFile (filter, private_data);
  g_assert (retval == 0);       /** This must be called only once */
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 */
static int
tf_invoke (const GstTensorFilter * filter, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  int retval;
  tf_data *tf;
  tf = *private_data;
  g_assert (filter->privateData && *private_data == filter->privateData);
  retval = tf_core_invoke (tf->tf_private_data, input, output);
  g_assert (retval == 0);
  return retval;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static int
tf_getInputDim (const GstTensorFilter * filter, void **private_data,
    GstTensorsInfo * info)
{
  int temp_idx = 0;
  tf_data *tf;
  tf = *private_data;
  temp_idx = tf_core_getInputSize (tf->tf_private_data);
  if (temp_idx > 0)
    temp_idx--;
  else
    temp_idx = 0;
  g_assert (filter->privateData && *private_data == filter->privateData);
  return tf_core_getInputDim (tf->tf_private_data, info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static int
tf_getOutputDim (const GstTensorFilter * filter, void **private_data,
    GstTensorsInfo * info)
{
  int temp_idx = 0;
  tf_data *tf;
  tf = *private_data;
  temp_idx = tf_core_getOutputSize (tf->tf_private_data);
  if (temp_idx > 0)
    temp_idx--;
  else
    temp_idx = 0;
  g_assert (filter->privateData && *private_data == filter->privateData);
  return tf_core_getOutputDim (tf->tf_private_data, info);
}

/**
 * @brief The set-input-dim callback for GstTensorFilterFramework
 */
static int
tf_setInputDim (const GstTensorFilter * filter, void **private_data,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  /** @todo call tflite core apis */
  return 0;                     /** NYI */
}

/**
 * @brief Free privateData and move on.
 */
static void
tf_close (const GstTensorFilter * filter, void **private_data)
{
  tf_data *tf;
  tf = *private_data;
  tf_core_delete (tf->tf_private_data);
  g_free (tf);
  *private_data = NULL;
  g_assert (filter->privateData == NULL);
}

GstTensorFilterFramework NNS_support_tensorflow = {
  .name = "tensorflow",
  .allow_in_place = FALSE,      /** @todo: support this to optimize performance later. */
  .allocate_in_invoke = TRUE,
  .invoke_NN = tf_invoke,
  .getInputDimension = tf_getInputDim,
  .getOutputDimension = tf_getOutputDim,
  .setInputDimension = tf_setInputDim,
  .open = tf_open,
  .close = tf_close,
};
