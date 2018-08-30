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
 * Fill in "GstTensor_Filter_Framework" for tensor_filter.h/c
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
tf_loadModelFile (const GstTensor_Filter * filter, void **private_data)
{
  tf_data *tf;
  if (filter->privateData != NULL) {
    /** @todo : Check the integrity of filter->data and filter->modelFilename, nnfw */
    return 1;
  }
  tf = g_new0 (tf_data, 1); /** initialize tf Fill Zero! */
  *private_data = tf;
  tf->tf_private_data = tf_core_new (filter->prop.modelFilename);
  if (tf->tf_private_data) {
    return 0;
  } else {
    return -1;
  }
}

/**
 * @brief The open callback for GstTensor_Filter_Framework. Called before anything else
 * @param filter : tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 */
static void
tf_open (const GstTensor_Filter * filter, void **private_data)
{
  int retval = tf_loadModelFile (filter, private_data);
  g_assert (retval == 0);       /** This must be called only once */
}

/**
 * @brief The mandatory callback for GstTensor_Filter_Framework
 * @param[in] inptr The input tensor
 * @param[out] outptr The output tensor
 */
static uint8_t *
tf_invoke (const GstTensor_Filter * filter, void **private_data,
    const uint8_t * inptr, uint8_t * outptr)
{
  int retval;
  uint8_t *allocated_outptr;
  tf_data *tf;
  tf = *private_data;
  g_assert (filter->privateData && *private_data == filter->privateData);
  retval =
      tf_core_invoke (tf->tf_private_data, (uint8_t *) inptr,
      &allocated_outptr);
  g_assert (retval == 0);
  return allocated_outptr;
}

/**
 * @brief The optional callback for GstTensor_Filter_Framework
 */
static int
tf_getInputDim (const GstTensor_Filter * filter, void **private_data,
    GstTensor_TensorsMeta * meta)
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
  return tf_core_getInputDim (tf->tf_private_data, meta->dims[0],
      &meta->types[0], &meta->num_tensors);
}

/**
 * @brief The optional callback for GstTensor_Filter_Framework
 */
static int
tf_getOutputDim (const GstTensor_Filter * filter, void **private_data,
    GstTensor_TensorsMeta * meta)
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
  return tf_core_getOutputDim (tf->tf_private_data, meta->dims[0],
      &meta->types[0], &meta->num_tensors);
}

/**
 * @brief The set-input-dim callback for GstTensor_Filter_Framework
 */
static int
tf_setInputDim (const GstTensor_Filter * filter, void **private_data,
    const tensor_dim iDimension, const tensor_type iType,
    tensor_dim oDimension, tensor_type * oType)
{
  /** @todo call tflite core apis */
  return 0;                     /** NYI */
}

/**
 * @brief Free privateData and move on.
 */
static void
tf_close (const GstTensor_Filter * filter, void **private_data)
{
  tf_data *tf;
  tf = *private_data;
  tf_core_delete (tf->tf_private_data);
  g_free (tf);
  *private_data = NULL;
  g_assert (filter->privateData == NULL);
}

GstTensor_Filter_Framework NNS_support_tensorflow = {
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
