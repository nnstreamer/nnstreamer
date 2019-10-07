/**
 * GStreamer Tensor_Filter, caffe2 Module
 * Copyright (C) 2019 Hyoung Joo Ahn <hello.ahn@samsung.com>
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
 * @file	tensor_filter_caffe2.c
 * @date	27 May 2019
 * @brief	Caffe2 for tensor_filter gstreamer plugin
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	HyoungJoo Ahn <hello.ahn@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (caffe2) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include <glib.h>
#include <string.h>

#include "tensor_filter_caffe2_core.h"

void init_filter_caffe2 (void) __attribute__ ((constructor));
void fini_filter_caffe2 (void) __attribute__ ((destructor));

/**
 * @brief internal data of caffe2
 */
struct _Caffe2_data
{
  void *caffe2_private_data;
};
typedef struct _Caffe2_data caffe2_data;


/**
 * @brief Free privateData and move on.
 */
static void
caffe2_close (const GstTensorFilterProperties * prop, void **private_data)
{
  caffe2_data *cf2;
  cf2 = *private_data;
  caffe2_core_delete (cf2->caffe2_private_data);
  g_free (cf2);
  *private_data = NULL;
}

/**
 * @brief Load caffe2 modelfile
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
caffe2_loadModelFile (const GstTensorFilterProperties * prop,
    void **private_data)
{
  caffe2_data *cf2;
  if (*private_data != NULL) {
    /** @todo : Check the integrity of filter->data and filter->model_file, nnfw */
    cf2 = *private_data;
    if (g_strcmp0 (prop->model_file,
            caffe2_core_getPredModelPath (cf2->caffe2_private_data)) != 0 ||
        g_strcmp0 (prop->model_file_sub,
            caffe2_core_getInitModelPath (cf2->caffe2_private_data)) != 0) {
      caffe2_close (prop, private_data);
    } else {
      return 1;
    }
  }
  cf2 = g_new0 (caffe2_data, 1); /** initialize cf2 Fill Zero! */
  *private_data = cf2;
  cf2->caffe2_private_data = caffe2_core_new (prop->model_file,
      prop->model_file_sub);
  if (cf2->caffe2_private_data) {
    if (caffe2_core_init (cf2->caffe2_private_data, prop)) {
      g_critical ("failed to initialize the object: Caffe2");
      g_free (cf2);
      return -2;
    }
    return 0;
  } else {
    g_critical ("failed to create the object: Caffe2");
    g_free (cf2);
    return -1;
  }
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 */
static int
caffe2_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int ret = caffe2_loadModelFile (prop, private_data);
  g_assert (ret >= 0);       /** This must be called only once */
  return ret;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
caffe2_run (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  int retval;
  caffe2_data *cf2;
  cf2 = *private_data;
  g_assert (*private_data);
  retval = caffe2_core_run (cf2->caffe2_private_data, input, output);
  g_assert (retval == 0);
  return retval;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static int
caffe2_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  caffe2_data *cf2;
  cf2 = *private_data;
  g_assert (*private_data);
  return caffe2_core_getInputDim (cf2->caffe2_private_data, info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static int
caffe2_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  caffe2_data *cf2;
  cf2 = *private_data;
  g_assert (*private_data);
  return caffe2_core_getOutputDim (cf2->caffe2_private_data, info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] data The data element.
 */
static void
caffe2_destroyNotify (void *data)
{
  caffe2_core_destroyNotify (data);
}

static gchar filter_subplugin_caffe2[] = "caffe2";

static GstTensorFilterFramework NNS_support_caffe2 = {
  .name = filter_subplugin_caffe2,
  .allow_in_place = FALSE,      /** @todo: support this to optimize performance later. */
  .allocate_in_invoke = TRUE,
  .destroyNotify = caffe2_destroyNotify,
  .invoke_NN = caffe2_run,
  .getInputDimension = caffe2_getInputDim,
  .getOutputDimension = caffe2_getOutputDim,
  .open = caffe2_open,
  .close = caffe2_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_caffe2 (void)
{
  nnstreamer_filter_probe (&NNS_support_caffe2);
}

/** @brief Destruct the subplugin */
void
fini_filter_caffe2 (void)
{
  nnstreamer_filter_exit (NNS_support_caffe2.name);
}
