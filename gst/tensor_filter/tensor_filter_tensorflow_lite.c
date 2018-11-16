/**
 * GStreamer Tensor_Filter, Tensorflow-Lite Module
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_filter_tensorflow_lite.c
 * @date	24 May 2018
 * @brief	Tensorflow-lite module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow-lite) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include "tensor_filter.h"
#include "tensor_filter_tensorflow_lite_core.h"
#include <glib.h>

/**
 * @brief internal data of tensorflow lite
 */
struct _Tflite_data
{
  void *tflite_private_data;
};
typedef struct _Tflite_data tflite_data;

/**
 * @brief Load tensorflow lite modelfile
 * @param filter : tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
tflite_loadModelFile (const GstTensorFilter * filter, void **private_data)
{
  tflite_data *tf;
  if (filter->privateData != NULL) {
    /** @todo : Check the integrity of filter->data and filter->model_file, nnfw */
    return 1;
  }
  tf = g_new0 (tflite_data, 1); /** initialize tf Fill Zero! */
  *private_data = tf;
  tf->tflite_private_data = tflite_core_new (filter->prop.model_file);
  if (tf->tflite_private_data) {
    if (tflite_core_init (tf->tflite_private_data))
      return -2;
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
static int
tflite_open (const GstTensorFilter * filter, void **private_data)
{
  int retval = tflite_loadModelFile (filter, private_data);
  g_assert (retval == 0);       /** This must be called only once */
  return 0;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
tflite_invoke (const GstTensorFilter * filter, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  int retval;
  tflite_data *tf;
  tf = *private_data;
  g_assert (filter->privateData && *private_data == filter->privateData);
  retval = tflite_core_invoke (tf->tflite_private_data, input, output);
  g_assert (retval == 0);
  return retval;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static int
tflite_getInputDim (const GstTensorFilter * filter, void **private_data,
    GstTensorsInfo * info)
{
  tflite_data *tf;
  tf = *private_data;
  g_assert (filter->privateData && *private_data == filter->privateData);
  int ret = tflite_core_getInputDim (tf->tflite_private_data, info);
  return ret;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static int
tflite_getOutputDim (const GstTensorFilter * filter, void **private_data,
    GstTensorsInfo * info)
{
  tflite_data *tf;
  tf = *private_data;
  g_assert (filter->privateData && *private_data == filter->privateData);
  int ret = tflite_core_getOutputDim (tf->tflite_private_data, info);
  return ret;
}

/**
 * @brief Free privateData and move on.
 */
static void
tflite_close (const GstTensorFilter * filter, void **private_data)
{
  tflite_data *tf;
  tf = *private_data;
  tflite_core_delete (tf->tflite_private_data);
  g_free (tf);
  *private_data = NULL;
  g_assert (filter->privateData == NULL);
}

GstTensorFilterFramework NNS_support_tensorflow_lite = {
  .name = "tensorflow-lite",
  .allow_in_place = FALSE,      /** @todo: support this to optimize performance later. */
  .allocate_in_invoke = FALSE,
  .invoke_NN = tflite_invoke,
  .getInputDimension = tflite_getInputDim,
  .getOutputDimension = tflite_getOutputDim,
  .open = tflite_open,
  .close = tflite_close,
};
