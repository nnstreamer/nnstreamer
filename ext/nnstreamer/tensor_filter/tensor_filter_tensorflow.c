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
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include <nnstreamer_plugin_api_filter.h>
#include "tensor_filter_tensorflow_core.h"
#include <glib.h>
#include <string.h>
#include <nnstreamer_conf.h>

/**
 * @brief internal data of tensorflow
 */
struct _Tf_data
{
  void *tf_private_data;
};
typedef struct _Tf_data tf_data;


/**
 * @brief Free privateData and move on.
 */
static void
tf_close (const GstTensorFilterProperties * prop, void **private_data)
{
  tf_data *tf;
  tf = *private_data;
  tf_core_delete (tf->tf_private_data);
  g_free (tf);
  *private_data = NULL;
}

/**
 * @brief Load tensorflow modelfile
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
tf_loadModelFile (const GstTensorFilterProperties * prop, void **private_data)
{
  tf_data *tf;
  gboolean tf_mem_optmz;

  if (*private_data != NULL) {
    tf = *private_data;
    if (strcmp (prop->model_file, tf_core_getModelPath (tf->tf_private_data))) {
      tf_close (prop, private_data);
    } else {
      return 1;
    }
  }

  tf_mem_optmz = nnsconf_get_value_bool (NNSCONF_VAL_TF_MEM_OPTMZ);

  tf = g_new0 (tf_data, 1); /** initialize tf Fill Zero! */
  *private_data = tf;
  tf->tf_private_data = tf_core_new (prop->model_file);

  if (tf->tf_private_data) {
    if (tf_core_init (tf->tf_private_data, prop, tf_mem_optmz)) {
      g_printerr ("failed to initailize the object: tensorflow");
      return -2;
    }
    return 0;
  } else {
    g_printerr ("failed to create the object: tensorflow");
    return -1;
  }
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 */
static int
tf_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int retval = tf_loadModelFile (prop, private_data);
  g_assert (retval == 0);       /** This must be called only once */
  return retval;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 */
static int
tf_run (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  int retval;
  tf_data *tf;
  tf = *private_data;
  g_assert (*private_data);
  retval = tf_core_run (tf->tf_private_data, input, output);
  g_assert (retval == 0);
  return retval;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static int
tf_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  tf_data *tf;
  tf = *private_data;
  g_assert (*private_data);
  return tf_core_getInputDim (tf->tf_private_data, info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static int
tf_getOutputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  tf_data *tf;
  tf = *private_data;
  g_assert (*private_data);
  return tf_core_getOutputDim (tf->tf_private_data, info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] data The data element.
 */
static void
tf_destroyNotify (void *data)
{
  tf_core_destroyNotify (data);
}

static gchar filter_subplugin_tensorflow[] = "tensorflow";

static GstTensorFilterFramework NNS_support_tensorflow = {
  .name = filter_subplugin_tensorflow,
  .allow_in_place = FALSE,      /** @todo: support this to optimize performance later. */
  .allocate_in_invoke = TRUE,
  .destroyNotify = tf_destroyNotify,
  .invoke_NN = tf_run,
  .getInputDimension = tf_getInputDim,
  .getOutputDimension = tf_getOutputDim,
  .open = tf_open,
  .close = tf_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
__attribute__ ((constructor))
     void init_filter_tf (void)
{
  tensor_filter_probe (&NNS_support_tensorflow);
}

/** @brief Destruct the subplugin */
__attribute__ ((destructor))
     void fini_filter_tf (void)
{
  tensor_filter_exit (NNS_support_tensorflow.name);
}
