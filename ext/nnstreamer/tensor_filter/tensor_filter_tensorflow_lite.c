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

#include <glib.h>
#include <string.h>

#include "tensor_filter_tensorflow_lite_core.h"
#include "tensor_common.h"

void init_filter_tflite (void) __attribute__ ((constructor));
void fini_filter_tflite (void) __attribute__ ((destructor));

/**
 * @brief internal data of tensorflow lite
 */
struct _Tflite_data
{
  void *tflite_private_data;
};
typedef struct _Tflite_data tflite_data;

/**
 * @brief nnapi hw type string
 */
static const char *nnapi_hw_string[] = {
  [NNAPI_CPU] = "cpu",
  [NNAPI_GPU] = "gpu",
  [NNAPI_NPU] = "npu",
  [NNAPI_UNKNOWN] = "unknown",
  NULL
};

/**
 * @brief return nnapi_hw type
 */
static nnapi_hw
get_nnapi_hw_type (const gchar * str)
{
  gint index = 0;
  index = find_key_strv (nnapi_hw_string, str);
  return (index < 0) ? NNAPI_UNKNOWN : index;
}

/**
 * @brief Free privateData and move on.
 */
static void
tflite_close (const GstTensorFilterProperties * prop, void **private_data)
{
  tflite_data *tf;
  tf = *private_data;
  tflite_core_delete (tf->tflite_private_data);
  g_free (tf);
  *private_data = NULL;
}

/**
 * @brief Load tensorflow lite modelfile
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
tflite_loadModelFile (const GstTensorFilterProperties * prop,
    void **private_data)
{
  tflite_data *tf;
  nnapi_hw hw = NNAPI_UNKNOWN;

  if (prop->nnapi) {
    gchar **strv = NULL;
    guint len;

    strv = g_strsplit (prop->nnapi, ":", 2);
    len = g_strv_length (strv);

    if (g_ascii_strcasecmp (strv[0], "true") == 0) {
      if (len >= 2) {
        hw = get_nnapi_hw_type (strv[1]);
      } else {
        /** defalut hw for nnapi is CPU */
        hw = NNAPI_CPU;
      }

      g_info ("NNAPI HW type: %s", nnapi_hw_string[hw]);
    }

    g_strfreev (strv);
  }

  if (*private_data != NULL) {
    /** @todo : Check the integrity of filter->data and filter->model_file, nnfw */
    tf = *private_data;
    if (g_strcmp0 (prop->model_file,
            tflite_core_getModelPath (tf->tflite_private_data)) != 0) {
      tflite_close (prop, private_data);
    } else {
      return 1;
    }
  }
  tf = g_new0 (tflite_data, 1); /** initialize tf Fill Zero! */
  *private_data = tf;
  tf->tflite_private_data = tflite_core_new (prop->model_file, hw);
  if (tf->tflite_private_data) {
    if (tflite_core_init (tf->tflite_private_data)) {
      g_printerr ("failed to initialize the object: Tensorflow-lite");
      return -2;
    }
    return 0;
  } else {
    g_printerr ("failed to create the object: Tensorflow-lite");
    return -1;
  }
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 */
static int
tflite_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int ret = tflite_loadModelFile (prop, private_data);
  g_assert (ret >= 0);       /** This must be called only once */
  return ret;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
tflite_invoke (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  int retval;
  tflite_data *tf;
  tf = *private_data;
  g_assert (*private_data);
  retval = tflite_core_invoke (tf->tflite_private_data, input, output);
  g_assert (retval == 0);
  return retval;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static int
tflite_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  tflite_data *tf;
  tf = *private_data;
  g_assert (*private_data);
  return tflite_core_getInputDim (tf->tflite_private_data, info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static int
tflite_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  tflite_data *tf;
  tf = *private_data;
  g_assert (*private_data);
  return tflite_core_getOutputDim (tf->tflite_private_data, info);
}

static gchar filter_subplugin_tensorflow_lite[] = "tensorflow-lite";

static GstTensorFilterFramework NNS_support_tensorflow_lite = {
  .name = filter_subplugin_tensorflow_lite,
  .allow_in_place = FALSE,      /** @todo: support this to optimize performance later. */
  .allocate_in_invoke = FALSE,
  .invoke_NN = tflite_invoke,
  .getInputDimension = tflite_getInputDim,
  .getOutputDimension = tflite_getOutputDim,
  .open = tflite_open,
  .close = tflite_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_tflite (void)
{
  nnstreamer_filter_probe (&NNS_support_tensorflow_lite);
}

/** @brief Destruct the subplugin */
void
fini_filter_tflite (void)
{
  nnstreamer_filter_exit (NNS_support_tensorflow_lite.name);
}
